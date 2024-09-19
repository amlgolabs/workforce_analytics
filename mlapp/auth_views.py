from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Profile  # Assuming you have a Profile model for mobile numbers

def custom_login(request):
    if request.method == 'POST':
        login_field = request.POST.get('username')  # Changed from 'login' to 'username'
        password = request.POST.get('password')

        if not login_field or not password:
            messages.error(request, 'Please provide both username/email/mobile and password')
            return render(request, 'registration/login.html')

        # Try to authenticate with username
        user = authenticate(request, username=login_field, password=password)

        # If authentication fails, try with email
        if user is None:
            try:
                user_obj = User.objects.get(email=login_field)
                user = authenticate(request, username=user_obj.username, password=password)
            except User.DoesNotExist:
                pass

        # If still not authenticated, try with mobile number
        if user is None:
            try:
                profile = Profile.objects.get(mobile=login_field)
                user = authenticate(request, username=profile.user.username, password=password)
            except Profile.DoesNotExist:
                pass

        if user is not None:
            login(request, user)
            return redirect('home')  # Replace 'home' with your home page URL name
        else:
            messages.error(request, 'Invalid login credentials')

    return render(request, 'registration/login.html')
