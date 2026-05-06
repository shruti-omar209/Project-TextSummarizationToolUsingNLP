from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from database import db

# =========================
# AUTH TABLE (Users + Admins)
# =========================
class User(UserMixin, db.Model):
    __tablename__ = "AUTH"

    user_id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    username = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        nullable=False,
        default="user"
    )  # 'user' or 'admin'

    registration_date = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    # Relationships
    activities = db.relationship("Activity", backref="user", lazy=True)
    inputs = db.relationship("InputContent", backref="user", lazy=True)
    summaries = db.relationship("Summary", backref="user", lazy=True)

    def get_id(self):
        return str(self.user_id)


# =========================
# ACTIVITY TABLE (Users + Admins)
# =========================
class Activity(db.Model):
    __tablename__ = "ACTIVITY"

    activity_id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("AUTH.user_id"),
        nullable=False
    )

    activity_type = db.Column(
        db.String(100),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        nullable=True
    )  # 'user' or 'admin'

    details = db.Column(
        db.Text,
        nullable=True
    )

    related_id = db.Column(
        db.Integer,
        nullable=True
    )

    activity_time = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow
    )


# =========================
# INPUT_CONTENT TABLE
# =========================
class InputContent(db.Model):
    __tablename__ = "INPUT_CONTENT"

    input_id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("AUTH.user_id"),
        nullable=False
    )

    input_type = db.Column(
        db.String(20),
        nullable=False
    )  # Text / PDF / TXT

    input_text = db.Column(
        db.Text,
        nullable=False
    )

    file_name = db.Column(
        db.String(255),
        nullable=True
    )

    summaries = db.relationship("Summary", backref="input", lazy=True)


# =========================
# SUMMARY TABLE
# =========================
class Summary(db.Model):
    __tablename__ = "SUMMARY"

    summary_id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("AUTH.user_id"),
        nullable=False
    )

    input_id = db.Column(
        db.Integer,
        db.ForeignKey("INPUT_CONTENT.input_id"),
        nullable=False
    )

    summary_text = db.Column(
        db.Text,
        nullable=False
    )

    summary_type = db.Column(
        db.String(20),
        nullable=False
    )  # Concise / Detailed