from flask import Blueprint, render_template, request
import os

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        # Placeholder: replace with ML model prediction
        result = f"Processed: {user_input}"
    return render_template("index.html", result=result)
