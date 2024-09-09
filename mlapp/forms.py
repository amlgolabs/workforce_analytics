from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Contact

class UserRegistrationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
        

class ContactForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = ['name', 'email', 'message']
        
class UploadFileForm(forms.Form):
    data_file = forms.FileField(label='Select a CSV or Excel file', help_text='max. 5 MB', 
                               widget=forms.FileInput(attrs={'accept': '.csv, .xlsx, .xls'}))        
        
        
# class UploadFileForm(forms.Form):
#     csv_file = forms.FileField(label='Select a CSV or Excel file', help_text='max. 5 MB', 
#                                widget=forms.FileInput(attrs={'accept': '.csv, .xlsx, .xls'}))
