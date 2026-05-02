from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import pipeline
import PyPDF2
from datetime import timedelta

from database import db, login_manager
from models import User, InputContent, Summary, UserActivity

app = Flask(__name__)

app.secret_key = "ai_summarizer_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager.login_view = "login"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


@app.route("/")
def index():
    return render_template("index.html")


# ---------------- REGISTER ----------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        hashed_password = generate_password_hash(password)

        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        activity = UserActivity(
            user_id=new_user.user_id,
            activity_type="User Registered"
        )

        db.session.add(activity)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):

            login_user(user)

            activity = UserActivity(
                user_id=user.user_id,
                activity_type="User Login"
            )

            db.session.add(activity)
            db.session.commit()

            return redirect(url_for("index"))

        return "Invalid credentials"

    return render_template("login.html")


# ---------------- LOGOUT ----------------

@app.route("/logout")
@login_required
def logout():

    activity = UserActivity(
        user_id=current_user.user_id,
        activity_type="User Logout"
    )

    db.session.add(activity)
    db.session.commit()

    logout_user()

    return redirect(url_for("index"))


# ---------------- GENERATE SUMMARY ----------------

@app.route("/generate", methods=["POST"])
def generate():

    text = request.form.get("text", "")
    summary_type = request.form.get("summary_type", "concise")

    if "pdf_file" in request.files and request.files["pdf_file"].filename != "":

        pdf = request.files["pdf_file"]
        reader = PyPDF2.PdfReader(pdf)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    words = text.split()
    text = " ".join(words[:500])

    original_words = len(text.split())

    percentage = 0.25 if summary_type == "concise" else 0.55
    target_words = int(original_words * percentage)

    max_len = max(40, int(target_words * 1.3))
    min_len = max(20, int(max_len * 0.6))

    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        truncation=True
    )

    summary = result[0]["summary_text"]

    return jsonify({
        "summary": summary,
        "word_count": len(summary.split()),
        "original_text": text
    })


# ---------------- SAVE SUMMARY ----------------

@app.route("/save_summary", methods=["POST"])
@login_required
def save_summary():

    data = request.get_json()

    text = data["text"]
    summary = data["summary"]
    summary_type = data["summary_type"]

    input_record = InputContent(
        user_id=current_user.user_id,
        input_type="Text",
        input_text=text,
        file_name=None
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

    activity = UserActivity(
        user_id=current_user.user_id,
        activity_type="Summary Generated",
        related_id=summary_record.summary_id
    )

    db.session.add(activity)
    db.session.commit()

    return jsonify({"status": "saved"})


# ---------------- HISTORY ----------------

@app.route("/history")
@login_required
def history():

    results = db.session.query(
        Summary,
        UserActivity.activity_time
    ).join(
        UserActivity,
        UserActivity.related_id == Summary.summary_id
    ).filter(
        Summary.user_id == current_user.user_id,
        UserActivity.activity_type == "Summary Generated"
    ).order_by(
        UserActivity.activity_time.desc()
    ).all()

    activity = UserActivity(
        user_id=current_user.user_id,
        activity_type="Viewed History"
    )

    db.session.add(activity)
    db.session.commit()

    return render_template("history.html", results=results, timedelta=timedelta)

# ---------------- DELETE SUMMARY ----------------

@app.route("/delete_summary/<int:summary_id>", methods=["POST"])
@login_required
def delete_summary(summary_id):

    summary = Summary.query.get_or_404(summary_id)

    # Security check (VERY IMPORTANT)
    if summary.user_id != current_user.user_id:
        return "Unauthorized", 403

    # Delete related input
    input_record = InputContent.query.get(summary.input_id)
    if input_record:
        db.session.delete(input_record)

    # Delete summary
    db.session.delete(summary)

    db.session.commit()

    return redirect(url_for("history"))

# ---------------- DELETE ALL SUMMARIES ----------------

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


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)