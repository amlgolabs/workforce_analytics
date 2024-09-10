# mlapp/models.py

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f'{self.name} - {self.email}'

class EmployeeData(models.Model):
    Age = models.FloatField()
    Attrition = models.CharField(max_length=255)
    BusinessTravel = models.CharField(max_length=255)
    DailyRate = models.FloatField()
    Department = models.CharField(max_length=255)
    DistanceFromHome = models.FloatField()
    Education = models.FloatField()
    EducationField = models.CharField(max_length=255)
    EmployeeCount = models.FloatField()
    EmployeeNumber = models.FloatField()
    EnvironmentSatisfaction = models.FloatField()
    Gender = models.CharField(max_length=255)
    HourlyRate = models.FloatField()
    JobInvolvement = models.FloatField()
    JobLevel = models.FloatField()
    JobRole = models.CharField(max_length=255)
    JobSatisfaction = models.FloatField()
    MaritalStatus = models.CharField(max_length=255)
    MonthlyIncome = models.FloatField()
    MonthlyRate = models.FloatField()
    NumCompaniesWorked = models.FloatField()
    Over18 = models.CharField(max_length=255)
    OverTime = models.CharField(max_length=255)
    PercentSalaryHike = models.FloatField()
    PerformanceRating = models.FloatField()
    RelationshipSatisfaction = models.FloatField()
    StandardHours = models.FloatField()
    StockOptionLevel = models.FloatField()
    TotalWorkingYears = models.FloatField()
    TrainingTimesLastYear = models.FloatField()
    WorkLifeBalance = models.FloatField()
    YearsAtCompany = models.FloatField()
    YearsInCurrentRole = models.FloatField()
    YearsSinceLastPromotion = models.FloatField()
    YearsWithCurrManager = models.FloatField()

    def __str__(self):
        return f'Employee {self.EmployeeNumber}'

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    mobile = models.CharField(max_length=15, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)

    def __str__(self):
        return self.user.username

