from flask import Flask, jsonify, request
import pyodbc
import random

app = Flask(__name__)

# Database connection configuration
server = 'your_server_name'
database = 'your_database_name'
username = 'your_username'
password = 'your_password'
driver = 'SQL Server'

# Database connection string
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Establish a database connection
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Create the RecordsComparisons and RecordsComparisonTagging tables if they don't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS RecordsComparisons (
        Record1ID INT NOT NULL,
        Record2ID INT NOT NULL,
        Name1 VARCHAR(50) NOT NULL,
        Name2 VARCHAR(50) NOT NULL,
        Age1 INT NOT NULL,
        Age2 INT NOT NULL,
        ProbabilityScore DOUBLE NOT NULL DEFAULT 0.0,
        IsTagged BIT NOT NULL DEFAULT 0,
        CONSTRAINT PK_RecordsComparisons PRIMARY KEY (Record1ID, Record2ID)
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS RecordsComparisonTagging (
        Record1ID INT NOT NULL,
        Record2ID INT NOT NULL,
        IsEqual BIT NOT NULL,
        Description VARCHAR(100),
        UserName VARCHAR(50) NOT NULL,
        ProbabilityScore DOUBLE NOT NULL DEFAULT 0.0,
        CONSTRAINT PK_RecordsComparisonTagging PRIMARY KEY (Record1ID, Record2ID)
    )
""")
conn.commit()


@app.route('/api/records-comparisons', methods=['GET'])
def get_records_comparisons():
    try:
        # Fetch all the records comparisons from the RecordsComparisons table
        query = "SELECT Record1ID, Record2ID, Name1, Name2, Age1, Age2, ProbabilityScore FROM RecordsComparisons"
        cursor.execute(query)
        rows = cursor.fetchall()

        # Prepare the response data
        records_comparisons = []
        for row in rows:
            record1_id, record2_id, name1, name2, age1, age2, probability_score = row
            record_comparison = {
                'record1_id': record1_id,
                'record2_id': record2_id,
                'name1': name1,
                'name2': name2,
                'age1': age1,
                'age2': age2,
                'probability_score': probability_score
            }
            records_comparisons.append(record_comparison)

        return jsonify(records_comparisons), 200

    except Exception as e:
        return f"Error retrieving records comparisons: {str(e)}", 500


@app.route('/api/records-comparison-tagging', methods=['POST'])
def save_record_comparison_tagging():
    try:
        # Get the tagging data from the request
        data = request.get_json()

        # Extract the fields from the data
        record1_id = data['record1_id']
        record2_id = data['record2_id']
        is_equal = data['is_equal']
        description = data['description']
        user_name = data['user_name']
        probability_score = data['probability_score']

        # Save the tagging in the RecordsComparisonTagging table
        query = "INSERT INTO RecordsComparisonTagging (Record1ID, Record2ID, IsEqual, Description, UserName, ProbabilityScore) VALUES (?, ?, ?, ?, ?, ?)"
        cursor.execute(query, record1_id, record2_id, is_equal, description, user_name, probability_score)
        conn.commit()

        # Update the isTagged field in the RecordsComparisons table
        update_query = "UPDATE RecordsComparisons SET IsTagged = 1 WHERE Record1ID = ? AND Record2ID = ?"
        cursor.execute(update_query, record1_id, record2_id)
        conn.commit()

        return 'Tagging saved successfully', 200

    except Exception as e:
        return f"Error saving record comparison tagging: {str(e)}", 500


@app.route('/api/not-tagged-record', methods=['GET'])
def get_not_tagged_record():
    try:
        # Fetch a random not-tagged record from the RecordsComparisons table
        query = "SELECT TOP 1 Record1ID, Record2ID, Name1, Name2, Age1, Age2, ProbabilityScore FROM RecordsComparisons WHERE IsTagged = 0 ORDER BY NEWID()"
        cursor.execute(query)
        row = cursor.fetchone()

        # Prepare the response data
        if row:
            record1_id, record2_id, name1, name2, age1, age2, probability_score = row
            record_comparison = {
                'record1_id': record1_id,
                'record2_id': record2_id,
                'name1': name1,
                'name2': name2,
                'age1': age1,
                'age2': age2,
                'probability_score': probability_score
            }
            return jsonify(record_comparison), 200
        else:
            return 'No untagged records found', 404

    except Exception as e:
        return f"Error retrieving not-tagged record: {str(e)}", 500


if __name__ == '__main__':
    app.run()




import pandas as pd
import pyodbc

def insert_records_from_excel(excel_file_path):
    # Database connection configuration
    server = 'your_server_name'
    database = 'your_database_name'
    username = 'your_username'
    password = 'your_password'
    driver = 'SQL Server'

    # Database connection string
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

    # Establish a database connection
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        record1_id = row['record1_id']
        record2_id = row['record2_id']
        name1 = row['name1']
        name2 = row['name2']
        age1 = row['age1']
        age2 = row['age2']

        # Insert the record into the RecordsComparisons table
        cursor.execute("""
            INSERT INTO RecordsComparisons (Record1ID, Record2ID, Name1, Name2, Age1, Age2)
            VALUES (?, ?, ?, ?, ?, ?)
        """, record1_id, record2_id, name1, name2, age1, age2)

    conn.commit()
    conn.close()

    print("Data inserted successfully into the RecordsComparisons table.")




import matplotlib.pyplot as plt
import pyodbc

def plot_tags_per_user():
    # Database connection configuration
    server = 'your_server_name'
    database = 'your_database_name'
    username = 'your_username'
    password = 'your_password'
    driver = 'SQL Server'

    # Database connection string
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

    # Establish a database connection
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Query the RecordsComparisonTagging table
    query = "SELECT UserName, COUNT(*) as TagCount FROM RecordsComparisonTagging GROUP BY UserName"
    cursor.execute(query)
    rows = cursor.fetchall()

    # Prepare the data for plotting
    usernames = []
    tag_counts = []
    for row in rows:
        username, tag_count = row
        usernames.append(username)
        tag_counts.append(tag_count)

    # Create a bar plot
    plt.bar(usernames, tag_counts)
    plt.xlabel("User Name")
    plt.ylabel("Tag Count")
    plt.title("Number of Tags per User")
    plt.show()

    conn.close()

# Call the function to generate the plot
plot_tags_per_user()



import random

def generate_sample_data():
    # Generate random sample data for testing
    usernames = ['User1', 'User2', 'User3', 'User4', 'User5']
    tag_counts = [random.randint(0, 20) for _ in range(len(usernames))]
    return usernames, tag_counts

def test_plot_tags_per_user():
    # Generate sample data
    usernames, tag_counts = generate_sample_data()

    # Create a bar plot using the sample data
    plt.bar(usernames, tag_counts)
    plt.xlabel("User Name")
    plt.ylabel("Tag Count")
    plt.title("Number of Tags per User")
    plt.show()

# Call the test function
test_plot_tags_per_user()





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_comparison_plot():
    # Fetch the comparison data from the database
    query = """
        SELECT ProbabilityScore, IsEqual
        FROM RecordsComparisonTagging
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    # Convert the SQL query result to a pandas DataFrame
    df = pd.DataFrame(rows, columns=['ProbabilityScore', 'IsEqual'])

    # Perform the segmentation using pandas
    df['ProbabilitySegment'] = pd.cut(df['ProbabilityScore'], bins=np.arange(0, 1.1, 0.1), right=False)

    # Calculate the count of equal and not equal comparisons for each segment
    segment_counts = df.groupby(['ProbabilitySegment', 'IsEqual']).size().unstack(fill_value=0)

    # Generate the bar plot
    x = np.arange(len(segment_counts))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, segment_counts[1], width, label='Equal')
    rects2 = ax.bar(x + width/2, segment_counts[0], width, label='Not Equal')

    # Set the plot labels and title
    ax.set_xlabel('Probability Score Segment')
    ax.set_ylabel('Comparison Count')
    ax.set_title('Comparison Count by Probability Score Segment')
    ax.set_xticks(x)
    ax.set_xticklabels(segment_counts.index)
    ax.legend()

    # Display the plot
    plt.show()



import matplotlib.pyplot as plt
import pandas as pd

def generate_comparison_plot1():
    # Fetch the comparison counts by probability score segmentation
    query = """
        SELECT
            FLOOR(ProbabilityScore / 0.10) * 0.10 AS ProbabilitySegment,
            SUM(CASE WHEN IsEqual = 1 THEN 1 ELSE 0 END) AS EqualCount,
            SUM(CASE WHEN IsEqual = 0 THEN 1 ELSE 0 END) AS NotEqualCount
        FROM RecordsComparisonTagging
        GROUP BY FLOOR(ProbabilityScore / 0.10)
        ORDER BY ProbabilitySegment
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    # Convert the SQL query result to a pandas DataFrame
    df = pd.DataFrame(rows, columns=['ProbabilitySegment', 'EqualCount', 'NotEqualCount'])

    # Generate the bar plot
    x = df['ProbabilitySegment']
    equal_counts = df['EqualCount']
    not_equal_counts = df['NotEqualCount']

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, equal_counts, label='Equal')
    rects2 = ax.bar(x, not_equal_counts, bottom=equal_counts, label='Not Equal')

    # Set the plot labels and title
    ax.set_xlabel('Probability Score Segment')
    ax.set_ylabel('Comparison Count')
    ax.set_title('Comparison Count by Probability Score Segment')
    ax.legend()

    # Display the plot
    plt.show()



import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flask import Flask, render_template, make_response

app = Flask(__name__)

@app.route('/comparison-plot', methods=['GET'])
def generate_comparison_plot():
    try:
        # Fetch the comparison data from the database
        query = """
            SELECT ProbabilityScore, IsEqual
            FROM RecordsComparisonTagging
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        # Convert the SQL query result to a pandas DataFrame
        df = pd.DataFrame(rows, columns=['ProbabilityScore', 'IsEqual'])

        # Perform the segmentation using pandas
        df['ProbabilitySegment'] = pd.cut(df['ProbabilityScore'], bins=np.arange(0, 1.1, 0.1), right=False)

        # Calculate the count of equal and not equal comparisons for each segment
        segment_counts = df.groupby(['ProbabilitySegment', 'IsEqual']).size().unstack(fill_value=0)

        # Generate the bar plot
        x = np.arange(len(segment_counts))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, segment_counts[1], width, label='Equal')
        rects2 = ax.bar(x + width/2, segment_counts[0], width, label='Not Equal')

        # Set the plot labels and title
        ax.set_xlabel('Probability Score Segment')
        ax.set_ylabel('Comparison Count')
        ax.set_title('Comparison Count by Probability Score Segment')
        ax.set_xticks(x)
        ax.set_xticklabels(segment_counts.index)
        ax.legend()

        # Convert the plot to a PNG image
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        plt.close(fig)

        # Create a response containing the image
        image_stream.seek(0)
        response = make_response(image_stream.getvalue())
        response.headers['Content-Type'] = 'image/png'
        return response

    except Exception as e:
        return f"Error generating comparison plot: {str(e)}", 500

if __name__ == '__main__':
    app.run()
