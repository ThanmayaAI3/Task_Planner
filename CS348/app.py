from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from sqlalchemy import text

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

class Task(db.Model):
    task_id = db.Column(db.Integer, primary_key=True)
    task_name = db.Column(db.String(200), nullable=False, index=True)
    due_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    planned_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id

@app.route('/')
def home():

    today = datetime.today()

    query = text("SELECT * FROM Task ORDER BY due_date")
    tasks = db.session.execute(query).fetchall()

    week_dates = [today + timedelta(days=i) for i in range(7)]

    # Create an empty dictionary to store tasks by date
    tasks_by_date = {date.strftime('%Y-%m-%d'): [] for date in week_dates}

    # Use a prepared statement to fetch tasks due within the range of today - 1 day to today + 6 days
    query = text("SELECT * FROM task WHERE due_date BETWEEN :start_date AND :end_date")
    tasks_this_week = db.session.execute(query, {
        'start_date': (today - timedelta(days=1)),
        'end_date': (today + timedelta(days=6))
    }).fetchall()

    # Group tasks by due_date
    for row in tasks_this_week:
        print(row.due_date)
        task = row.task_name  # Convert each row to a dictionary
        #datetime.strptime(row.due_date)
        date_str = datetime.strptime(row.due_date, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d')
        tasks_by_date[date_str].append(task)
    print("new code")
    print(tasks_by_date)
    ### END OF NEW

    query = text("SELECT * FROM task WHERE DATE(planned_date) = DATE(:today)")
    planned_today = db.session.execute(query, {'today': today}).fetchall()

    return render_template('index.html', tasks=tasks, week_dates=week_dates, tasks_by_date=tasks_by_date,
                           planned_today=planned_today)


@app.route('/add', methods=['POST'])
def add_task():
    task_name = request.form['task_name']
    planned_date = datetime.strptime(request.form['planned_date'], '%Y-%m-%d')
    due_date = datetime.strptime(request.form['due_date'], '%Y-%m-%d') if request.form['due_date'] else None

    new_task = Task(task_name=task_name, planned_date=planned_date, due_date=due_date)
    db.session.add(new_task)
    db.session.commit()
    return redirect(url_for('home'))

@app.route('/delete/<int:task_id>', methods=['GET'])
def delete_task(task_id):
    # Query for the task by ID
    task_to_delete = Task.query.get_or_404(task_id)
    try:
        # Delete the task
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect(url_for('home'))
    except:
        # Handle error if deletion fails
        return "There was a problem deleting the task."


@app.route('/update/<int:task_id>', methods=['GET', 'POST'])
def update_task(task_id):
    # Query for the task by ID
    task = Task.query.get_or_404(task_id)

    if request.method == 'POST':
        # Update the task's attributes from form data
        task.task_name = request.form['task_name']
        task.planned_date = datetime.strptime(request.form['planned_date'], '%Y-%m-%d')
        task.due_date = datetime.strptime(request.form['due_date'], '%Y-%m-%d')

        try:
            # Commit the changes
            db.session.commit()
            return redirect(url_for('home'))
        except:
            return "There was a problem updating the task."

    return render_template('update.html', task=task)


if __name__ == "__main__":
    app.run(debug=True)
