from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import requests
import PyPDF2
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from database import db, login_manager
from models import User, InputContent, Summary, Activity

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "fallback_secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Neon PostgreSQL drops idle connections — these settings auto-reconnect
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,       # test connection before use, reconnects if dropped
    "pool_recycle": 300,         # recycle connections every 5 minutes
    "pool_size": 5,
    "max_overflow": 2,
    "connect_args": {"connect_timeout": 10},
}

db.init_app(app)

login_manager.login_view = "login"
login_manager.init_app(app)

ALLOWED_EXTENSIONS = {"pdf", "txt"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Hugging Face Inference API
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def call_summarizer(text, max_length, min_length):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "truncation": True
        }
    }
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()[0]["summary_text"]


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def log_activity(user_id, role, activity_type, details=None, related_id=None):
    activity = Activity(
        user_id=user_id,
        role=role,
        activity_type=activity_type,
        details=details,
        related_id=related_id
    )
    db.session.add(activity)
    db.session.commit()


# ================================================================
# AUTH ROUTES
# ================================================================

@app.route("/")
def index():
    if current_user.is_authenticated and current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not email or not password:
            error = "All fields are required."
        elif User.query.filter_by(email=email).first():
            error = "Email already registered."
        elif User.query.filter_by(username=username).first():
            error = "Username already taken."
        else:
            new_user = User(
                username=username,
                email=email,
                password=generate_password_hash(password),
                role="user"
            )
            db.session.add(new_user)
            db.session.commit()
            log_activity(new_user.user_id, "user", "User Registered",
                         details=f"New user '{username}' registered")
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            error = "Username and password are required."
        else:
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                log_activity(user.user_id, user.role,
                             f"{'Admin' if user.role == 'admin' else 'User'} Login",
                             details=f"'{username}' logged in as {user.role}")
                if user.role == "admin":
                    return redirect(url_for("admin_dashboard"))
                return redirect(url_for("index"))
            else:
                error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
@login_required
def logout():
    log_activity(current_user.user_id, current_user.role,
                 f"{'Admin' if current_user.role == 'admin' else 'User'} Logout",
                 details=f"'{current_user.username}' logged out")
    logout_user()
    return redirect(url_for("login"))


# ================================================================
# MANAGE ACCOUNT
# ================================================================

@app.route("/account", methods=["GET", "POST"])
@login_required
def manage_account():
    error = None
    success = None
    if request.method == "POST":
        new_username = request.form.get("username", "").strip()
        new_email = request.form.get("email", "").strip()
        new_password = request.form.get("password", "").strip()

        if not new_username or not new_email:
            error = "Username and email cannot be empty."
        else:
            existing = User.query.filter(
                User.email == new_email, User.user_id != current_user.user_id
            ).first()
            if existing:
                error = "Email already in use."
            else:
                current_user.username = new_username
                current_user.email = new_email
                if new_password:
                    current_user.password = generate_password_hash(new_password)
                db.session.commit()
                log_activity(current_user.user_id, current_user.role,
                             "Account Updated",
                             details=f"User updated account details")
                success = "Account updated successfully."

    return render_template("manage_account.html", error=error, success=success)


# ================================================================
# SUMMARIZER ROUTES
# ================================================================

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    text = request.form.get("input_text", "")
    summary_type = request.form.get("summary_type", "Concise")
    input_type = "Text"

    if "file" in request.files and request.files["file"].filename != "":
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename

        if not allowed_file(filename):
            return jsonify({"error": "Only PDF and TXT files are allowed."}), 400

        ext = filename.rsplit(".", 1)[1].lower()
        input_type = ext.upper()

        if ext == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        elif ext == "txt":
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        log_activity(current_user.user_id, current_user.role,
                     "File Uploaded", details=f"Uploaded file: {filename}")

    if not text.strip():
        return jsonify({"error": "No text provided."}), 400

    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    words = text.split()
    text = " ".join(words[:500])
    original_words = len(text.split())

    percentage = 0.25 if summary_type == "concise" else 0.55
    target_words = int(original_words * percentage)
    max_len = max(40, int(target_words * 1.3))
    min_len = max(20, int(max_len * 0.6))

    summary = call_summarizer(text, max_len, min_len)

    return jsonify({
        "summary": summary,
        "word_count": len(summary.split()),
        "original_text": text,
        "input_type": input_type
    })


@app.route("/save_summary", methods=["POST"])
@login_required
def save_summary():
    # Accept both form data and JSON
    if request.is_json:
        data = request.get_json()
        text = data.get("input_text", "")
        summary = data.get("summary_text", "")
        summary_type = data.get("summary_type", "Concise")
        input_type = data.get("input_type", "Text")
        file_name = data.get("file_name", None)
    else:
        text = request.form.get("input_text", "")
        summary = request.form.get("summary_text", "")
        summary_type = request.form.get("summary_type", "Concise")
        input_type = request.form.get("input_type", "Text")
        file_name = request.form.get("file_name") or None

    if not summary.strip():
        return jsonify({"error": "No summary to save."}), 400

    input_record = InputContent(
        user_id=current_user.user_id,
        input_type=input_type,
        input_text=text,
        file_name=file_name
    )
    db.session.add(input_record)
    db.session.commit()

    summary_record = Summary(
        user_id=current_user.user_id,
        input_id=input_record.input_id,
        summary_text=summary,
        summary_type=summary_type
    )
    db.session.add(summary_record)
    db.session.commit()

    log_activity(current_user.user_id, current_user.role,
                 "Summary Generated",
                 details=f"Summary type: {summary_type}",
                 related_id=summary_record.summary_id)

    return jsonify({"status": "saved"})


@app.route("/history")
@login_required
def history():
    results = db.session.query(
        Summary, Activity.activity_time
    ).join(
        Activity, Activity.related_id == Summary.summary_id
    ).filter(
        Summary.user_id == current_user.user_id,
        Activity.activity_type == "Summary Generated"
    ).order_by(Activity.activity_time.desc()).all()

    log_activity(current_user.user_id, current_user.role, "Viewed History")

    return render_template("history.html", results=results, timedelta=timedelta)


@app.route("/delete_summary/<int:summary_id>", methods=["POST"])
@login_required
def delete_summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    if summary.user_id != current_user.user_id:
        return "Unauthorized", 403

    input_record = InputContent.query.get(summary.input_id)
    if input_record:
        db.session.delete(input_record)
    db.session.delete(summary)
    db.session.commit()
    return redirect(url_for("history"))


@app.route("/delete_all", methods=["POST"])
@login_required
def delete_all():
    summaries = Summary.query.filter_by(user_id=current_user.user_id).all()
    for s in summaries:
        input_record = InputContent.query.get(s.input_id)
        if input_record:
            db.session.delete(input_record)
        db.session.delete(s)
    db.session.commit()
    return redirect(url_for("history"))


# ================================================================
# ADMIN MODULE
# ================================================================

@app.route("/admin/dashboard")
@login_required
@admin_required
def admin_dashboard():
    total_users = User.query.filter_by(role="user").count()
    total_summaries = Summary.query.count()
    recent_logs = Activity.query.order_by(Activity.activity_time.desc()).limit(10).all()
    return render_template("admin_dashboard.html",
                           total_users=total_users,
                           total_summaries=total_summaries,
                           recent_logs=recent_logs)


@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    users = User.query.filter_by(role="user").order_by(User.registration_date.desc()).all()
    return render_template("admin_users.html", users=users)


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    uname = user.username
    for summary in user.summaries:
        input_record = InputContent.query.get(summary.input_id)
        if input_record:
            db.session.delete(input_record)
        db.session.delete(summary)
    Activity.query.filter_by(user_id=user_id).delete()
    db.session.delete(user)

    log = Activity(action="Delete User", details=f"Deleted user '{uname}' (ID: {user_id})")
    db.session.add(log)
    log_activity(current_user.user_id, "admin", "Admin: Delete User",
                 details=f"Admin deleted user '{uname}'")
    db.session.commit()

    flash(f"User '{uname}' deleted.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/update", methods=["POST"])
@login_required
@admin_required
def admin_update_user(user_id):
    user = User.query.get_or_404(user_id)
    new_username = request.form.get("username", "").strip()
    new_email = request.form.get("email", "").strip()

    if not new_username or not new_email:
        flash("Username and email cannot be empty.", "error")
        return redirect(url_for("admin_users"))

    if User.query.filter(User.email == new_email, User.user_id != user_id).first():
        flash("Email already in use.", "error")
        return redirect(url_for("admin_users"))

    user.username = new_username
    user.email = new_email
    log = Activity(action="Update User",
                        details=f"Updated user ID {user_id} → '{new_username}', '{new_email}'")
    db.session.add(log)
    log_activity(current_user.user_id, "admin", "Admin: Update User",
                 details=f"Updated user ID {user_id}")
    db.session.commit()
    flash("User updated.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/summaries")
@login_required
@admin_required
def admin_summaries():
    summaries = db.session.query(Summary, User.username).join(
        User, User.user_id == Summary.user_id
    ).order_by(Summary.summary_id.desc()).all()
    return render_template("admin_summaries.html", summaries=summaries)


@app.route("/admin/summaries/<int:summary_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    input_record = InputContent.query.get(summary.input_id)
    if input_record:
        db.session.delete(input_record)
    Activity.query.filter_by(
        activity_type="Summary Generated", related_id=summary_id
    ).delete()
    db.session.delete(summary)

    log = Activity(action="Delete Summary", details=f"Deleted summary ID {summary_id}")
    db.session.add(log)
    log_activity(current_user.user_id, "admin", "Admin: Delete Summary",
                 details=f"Deleted summary ID {summary_id}")
    db.session.commit()
    flash("Summary deleted.", "success")
    return redirect(url_for("admin_summaries"))


@app.route("/admin/logs")
@login_required
@admin_required
def admin_logs():
    logs = Activity.query.order_by(Activity.activity_time.desc()).all()
    return render_template("admin_logs.html", logs=logs)


with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print(f"[WARNING] Could not connect to DB on startup: {e}")


@app.cli.command("seed-admin")
def seed_admin():
    """Create admin user in DB. Run once: flask seed-admin"""
    username = input("Admin username: ")
    email = input("Admin email: ")
    password = input("Admin password: ")
    if User.query.filter_by(username=username).first():
        print("Username already exists.")
        return
    user = User(username=username, email=email,
                password=generate_password_hash(password), role="admin")
    db.session.add(user)
    db.session.commit()
    print(f"Admin '{username}' created successfully.")


if __name__ == "__main__":
    app.run(debug=True)
