# eq_website

Repository for earthquake prediction website.

## Quickstart

1. Download project, set up git and virtual environment
   ```
   git clone https://github.com/Michal1337/eq_website.git
   cd eq_website
   python -m venv venv
   source venv/Scripts/activate
   ```
2. Download requirements
    ```
    pip install -r requirements.txt
    ```
3. Follow instructions under [this link](https://docs.djangoproject.com/en/4.2/ref/contrib/gis/install) to download PostgreSQL and additional spatial libraries.
4. During the database setup install `OSGeo4W` in `C:\OSGeo4W`. Name the database `website_db`, user `postgres` and set the password to `postgres`.
5. Go to `website` directory, make and apply migrations
    ```
    cd website
    python manage.py makemigrations
    python manage.py migrate
    ```
6. Create superuser
    ```
   python manage.py createsuperuser
    ```
   You don't need to specify your email address, only username and password.
7. Load initial data
   ```
   python manage.py load_countries
   python manage.py load_earthquakes
   python manage.py load_models
   ```
8. Run server
   ```
   python manage.py runserver
   ```
   You can now view the website on `http://127.0.0.1:8000/`
9. To stop the server hit `ctrl+C` on the terminal
