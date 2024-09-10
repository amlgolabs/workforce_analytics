"""
URL configuration for ml_web_app project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import path
from mlapp import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.register, name='register'),
    path('services/', views.services, name='services'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('contact/success/', views.contact_success, name='contact_success'),
    path('data_statistics/', views.data_statistics, name='data_statistics'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('update_profile/', views.update_profile, name='update_profile'),
    path('clean_data/', views.clean_data, name='clean_data'),
    path('generate_bivariate_plot/', views.generate_bivariate_plot, name='generate_bivariate_plot'),
    path('detailed_eda/', views.detailed_eda, name='detailed_eda'),
    path('generate_multivariate_plot/', views.generate_multivariate_plot, name='generate_multivariate_plot'),
    path('advanced_analysis/', views.advanced_analysis, name='advanced_analysis'),
    path('predictions/', views.predictions, name='predictions'),
    path('ai/', views.ai, name='ai'),
    path('ai/run_code/', views.run_code, name='run_code'),
    path('ai/filter/', views.ai_filter, name='ai_filter'),
    path('ai/graph/', views.ai_graph, name='ai_graph'),
    path('predict/', views.predict_attrition, name='predict_attrition'),
    path('process_csv/', views.process_csv, name='process_csv'),
    path('process_csv2/', views.process_csv2, name='process_csv2'),
    path('predict_salary_hike/', views.predict_salary_hike, name='predict_salary_hike'),
    path('performance_prediction/', views.performance_prediction, name='performance_prediction'),
    path('process_csv3/', views.process_csv3, name='process_csv3'),
    path('generate_correlation_plot/', views.generate_correlation_plot, name='generate_correlation_plot'),
    path('download_result/', views.download_result, name='download_result'),
]
urlpatterns += staticfiles_urlpatterns()