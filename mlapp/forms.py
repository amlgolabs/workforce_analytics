from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Contact, Profile

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=False, help_text='Optional')
    mobile = forms.CharField(max_length=15, required=False, help_text='Optional')
    profile_picture = forms.ImageField(required=False, help_text='Optional')

    class Meta:
        model = User
        fields = ['username', 'email', 'mobile', 'password1', 'password2', 'profile_picture']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            profile = Profile.objects.create(user=user)
            profile.mobile = self.cleaned_data['mobile']
            if self.cleaned_data['profile_picture']:
                profile.profile_picture = self.cleaned_data['profile_picture']
            profile.save()

        return user

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

class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email']

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['mobile', 'profile_picture']

class PasswordChangeForm(forms.Form):
    old_password = forms.CharField(widget=forms.PasswordInput())
    new_password1 = forms.CharField(widget=forms.PasswordInput())
    new_password2 = forms.CharField(widget=forms.PasswordInput())

    def clean(self):
        cleaned_data = super().clean()
        new_password1 = cleaned_data.get('new_password1')
        new_password2 = cleaned_data.get('new_password2')

        if new_password1 and new_password2 and new_password1 != new_password2:
            raise forms.ValidationError("The two password fields didn't match.")
