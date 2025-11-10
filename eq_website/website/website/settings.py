# See
# https://docs.djangoproject.com/en/4.2/topics/settings/
# https://docs.djangoproject.com/en/4.2/ref/settings/

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-2_-(vf#7qur&+=4q(jgz5*&m4n5ekg1c%#ry52-+ni4x*e7gae'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['127.0.0.1']

# use this if setting up on Windows 10 with GDAL installed from OSGeo4W using defaults
# if os.name == 'nt':
#    VIRTUAL_ENV_BASE = os.environ.get('VIRTUAL_ENV')
#    os.environ['PATH'] = os.path.join(VIRTUAL_ENV_BASE, r'.\\Lib\\site-packages\\osgeo') + ';' + os.environ['PATH']
#    os.environ['PROJ_LIB'] = os.path.join(VIRTUAL_ENV_BASE, r'.\\Lib\\site-packages\\osgeo\\data\\proj') + ';' + os.environ['PATH']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',
    'rest_framework',
    'bootstrap5',
    'fontawesomefree',
    'map'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'website.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'website.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': 'website_db',
        'USER': 'postgres',
        'HOST': 'localhost',
        'PASSWORD': 'postgres',
        'PORT': '5432',
    }
}

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'
STATICFILES_DIRS = ((os.path.join(BASE_DIR, 'static')),)
# STATIC_ROOT = (os.path.join(BASE_DIR, 'static'))

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Media
MEDIA_URL = 'media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# user specific!
GDAL_LIBRARY_PATH = "C:\\OSGeo4W\\bin\\gdal308.dll"
GEOS_LIBRARY_PATH = "C:\\OSGeo4W\\bin\\geos_c.dll"
# PROJ_LIB_PATH = "C:\\OSGeo4W\\share\\proj"
# Ustawienie PROJ_LIB na podstawie ścieżki
# PROJ_LIB = os.path.join(PROJ_LIB_PATH, "proj")
