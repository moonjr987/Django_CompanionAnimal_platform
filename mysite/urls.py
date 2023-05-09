"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from pybo import views
from rest_framework import routers
from pybo.views import MovieViewSet
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

router = routers.DefaultRouter() 
router.register('movies',MovieViewSet) # prefix = movies , viewset = MovieViewSet

urlpatterns = [
    path("pybo/", include('pybo.urls')),
    #path('pybo/', include('allauth.urls')),
    path('pybo/',include('django.contrib.auth.urls')),
    #path('accounts/',include('django.contrib.auth.urls')),
    #path('accounts/',include('allauth.urls')),
    #re_path(r'^admin/', admin.site.urls),
    re_path(r'^api/',include(router.urls)),
    path("admin/", admin.site.urls),
    
    path('common/', include('common.urls')),
    path('', views.main, name='main'),  # '/' 에 해당되는 path
    
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
