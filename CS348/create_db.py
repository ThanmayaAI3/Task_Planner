from app import app, db  # Replace `your_application_module` with the actual module name

with app.app_context():
    db.create_all()