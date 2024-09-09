from django.shortcuts import render, redirect
from .models import Contact, EmployeeData
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
from .forms import UserRegistrationForm,ContactForm,UploadFileForm
from django.urls import reverse_lazy 
from django.http import HttpResponseServerError,JsonResponse,HttpResponse
import logging
import json
from django.utils import timezone
current_time = timezone.now()
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from sklearn.impute import KNNImputer

import joblib
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import glob
import io
import math
import pandas as pd
from IPython.display import display
from django.conf import settings
import itertools 
from statsmodels.graphics.mosaicplot import mosaic



logger = logging.getLogger(__name__)

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        print(current_time)
        if form.is_valid():
            form.save() 
            return redirect('contact_success') 
    else:
        form = ContactForm()
    return render(request, 'mlapp/contact.html', {'form': form})

def contact_success(request):
    return render(request, 'mlapp/contact_success.html')

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')  
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'registration_form': form})

def home(request):
    login_form = AuthenticationForm(request, request.POST or None)
    registration_form = UserRegistrationForm(request.POST or None)

    if request.method == 'POST':
        if 'login_form_submit' in request.POST:
            username = request.POST['username']
            password = request.POST['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return render(request, 'mlapp/home.html')

            else:
                messages.error(request, 'Invalid username or password.')

        elif 'registration_form_submit' in request.POST:
            if registration_form.is_valid():
                registration_form.save()
                return redirect('home')

    # If not a POST request or login not successful, render the template with login and registration forms
    return render(request, 'mlapp/home.html', {'login_form': login_form, 'registration_form': registration_form})

def services(request):
    return render(request, 'mlapp/services.html')

def about(request):
    return render(request, 'mlapp/about.html')




def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to home page after login
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'registration/login.html')

def user_logout(request):
    logout(request)
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    file_path2 = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file0.csv')
    file_path3 = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'cleaned_dataset.csv')
    try:
        # Read the csv file
        df = pd.read_csv(file_path2)
        
        # Save the dataframe to a csv file
        df.to_csv(file_path, index=False)
        df.to_csv(file_path3, index=False)
            # Data Validation: (Customize as needed)
            # Clear existing data 
        EmployeeData.objects.all().delete() 
            # Iterate over DataFrame rows and create model instances
        for _, row in df.iterrows():
                EmployeeData.objects.create(
                    Age=row.get('Age', None),
                    Attrition=row.get('Attrition', None),
                    BusinessTravel=row.get('BusinessTravel', None),
                    DailyRate=row.get('DailyRate', None),
                    Department=row.get('Department', None),
                    DistanceFromHome=row.get('DistanceFromHome', None),
                    Education=row.get('Education', None),
                    EducationField=row.get('EducationField', None),
                    EmployeeCount=row.get('EmployeeCount', None),
                    EmployeeNumber=row.get('EmployeeNumber', None),
                    EnvironmentSatisfaction=row.get('EnvironmentSatisfaction', None),
                    Gender=row.get('Gender', None),
                    HourlyRate=row.get('HourlyRate', None),
                    JobInvolvement=row.get('JobInvolvement', None),
                    JobLevel=row.get('JobLevel', None),
                    JobRole=row.get('JobRole', None),
                    JobSatisfaction=row.get('JobSatisfaction', None),
                    MaritalStatus=row.get('MaritalStatus', None),
                    MonthlyIncome=row.get('MonthlyIncome', None),
                    MonthlyRate=row.get('MonthlyRate', None),
                    NumCompaniesWorked=row.get('NumCompaniesWorked', None),
                    Over18=row.get('Over18', None),
                    OverTime=row.get('OverTime', None),
                    PercentSalaryHike=row.get('PercentSalaryHike', None),
                    PerformanceRating=row.get('PerformanceRating', None),
                    RelationshipSatisfaction=row.get('RelationshipSatisfaction', None),
                    StandardHours=row.get('StandardHours', None),
                    StockOptionLevel=row.get('StockOptionLevel', None),
                    TotalWorkingYears=row.get('TotalWorkingYears', None),
                    TrainingTimesLastYear=row.get('TrainingTimesLastYear', None),
                    WorkLifeBalance=row.get('WorkLifeBalance', None),
                    YearsAtCompany=row.get('YearsAtCompany', None),
                    YearsInCurrentRole=row.get('YearsInCurrentRole', None),
                    YearsSinceLastPromotion=row.get('YearsSinceLastPromotion', None),
                    YearsWithCurrManager=row.get('YearsWithCurrManager', None),)
        # Create images directory if it doesn't exist
        images_dir = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'images')
        os.makedirs(images_dir, exist_ok=True)
        # Generate and save plots
        plot_numerical_cols(df)
        plot_categorical_cols(df)
        return redirect('login')
    except Exception as e:
        # Handle any exceptions that occur during file operations
        print(f"An error occurred: {e}")
    

def handle_uploaded_file(f):
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

#file uploading function
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'} 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['data_file']

            if not allowed_file(uploaded_file.name):
                return HttpResponse("Invalid file type. Allowed types: csv, xlsx, xls")

            try:
                # Save uploaded file to a temporary location
                file_ext = uploaded_file.name.rsplit('.', 1)[1].lower()
                temp_file_path = os.path.join(settings.MEDIA_ROOT, 'temp_file.' + file_ext)
                
                with open(temp_file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                # Load DataFrame based on file type
                if file_ext == 'csv':
                    df = pd.read_csv(temp_file_path)
                elif file_ext in {'xlsx', 'xls'}:
                    df = pd.read_excel(temp_file_path)
                else:
                    os.remove(temp_file_path)
                    return HttpResponse("Unsupported file format.")

                # Save DataFrame to a permanent location
                data_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
                df.to_csv(data_file_path, index=False)
                df.to_csv(os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'cleaned_dataset.csv'),index=False)
                # Clear existing data
                EmployeeData.objects.all().delete()

                # Iterate over DataFrame rows and create model instances
                for _, row in df.iterrows():
                    EmployeeData.objects.create(
                        Age=row.get('Age', None),
                        Attrition=row.get('Attrition', None),
                        BusinessTravel=row.get('BusinessTravel', None),
                        DailyRate=row.get('DailyRate', None),
                        Department=row.get('Department', None),
                        DistanceFromHome=row.get('DistanceFromHome', None),
                        Education=row.get('Education', None),
                        EducationField=row.get('EducationField', None),
                        EmployeeCount=row.get('EmployeeCount', None),
                        EmployeeNumber=row.get('EmployeeNumber', None),
                        EnvironmentSatisfaction=row.get('EnvironmentSatisfaction', None),
                        Gender=row.get('Gender', None),
                        HourlyRate=row.get('HourlyRate', None),
                        JobInvolvement=row.get('JobInvolvement', None),
                        JobLevel=row.get('JobLevel', None),
                        JobRole=row.get('JobRole', None),
                        JobSatisfaction=row.get('JobSatisfaction', None),
                        MaritalStatus=row.get('MaritalStatus', None),
                        MonthlyIncome=row.get('MonthlyIncome', None),
                        MonthlyRate=row.get('MonthlyRate', None),
                        NumCompaniesWorked=row.get('NumCompaniesWorked', None),
                        Over18=row.get('Over18', None),
                        OverTime=row.get('OverTime', None),
                        PercentSalaryHike=row.get('PercentSalaryHike', None),
                        PerformanceRating=row.get('PerformanceRating', None),
                        RelationshipSatisfaction=row.get('RelationshipSatisfaction', None),
                        StandardHours=row.get('StandardHours', None),
                        StockOptionLevel=row.get('StockOptionLevel', None),
                        TotalWorkingYears=row.get('TotalWorkingYears', None),
                        TrainingTimesLastYear=row.get('TrainingTimesLastYear', None),
                        WorkLifeBalance=row.get('WorkLifeBalance', None),
                        YearsAtCompany=row.get('YearsAtCompany', None),
                        YearsInCurrentRole=row.get('YearsInCurrentRole', None),
                        YearsSinceLastPromotion=row.get('YearsSinceLastPromotion', None),
                        YearsWithCurrManager=row.get('YearsWithCurrManager', None),
                    )

                # Generate and save plots
                numerical_fig = plot_numerical_cols(df)
                categorical_fig = plot_categorical_cols(df)

                # Ensure the images directory exists
                images_dir = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'images')
                os.makedirs(images_dir, exist_ok=True)

                numerical_image_path = os.path.join(images_dir, 'numerical_plot.png')
                categorical_image_path = os.path.join(images_dir, 'categorical_plot.png')
                
                save_plot_as_image(numerical_fig, numerical_image_path)
                save_plot_as_image(categorical_fig, categorical_image_path)

                # Remove temporary file
                os.remove(temp_file_path)

                # Redirect or render as needed
                return redirect('home')

            except Exception as e:
                return HttpResponse(f"Error processing file: {str(e)}")

    else:
        form = UploadFileForm()
    return render(request, 'mlapp/upload_file.html', {'upload_form': form})

#detailed_eda part:

def plot_numerical_cols(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    n = 3
    m = math.ceil(len(numerical_cols) / n)
    
    fig, axes = plt.subplots(m, n, figsize=(5*n, 5*m))
    axes = axes.flatten() if m > 1 and n > 1 else [axes]
    
    for i, column in enumerate(numerical_cols):
        sns.histplot(df[column], kde=True, ax=axes[i])
        axes[i].text(0.02, 0.95, f'{i+1}', transform=axes[i].transAxes,
                     fontsize=15, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))
        axes[i].set_title(column, fontsize=15)
        axes[i].set_xlabel(column, fontsize=15)
        axes[i].set_ylabel('Frequency', fontsize=15)

    for j in range(len(numerical_cols), m * n):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def plot_categorical_cols(df):
    categorical_cols = df.select_dtypes(include=[object]).columns
    n = 3
    m = math.ceil(len(categorical_cols) / n)
    
    fig, axs = plt.subplots(nrows=m, ncols=n, figsize=(5*n, 7*m))

    for i, col in enumerate(categorical_cols[:n*m]):
        row = i // n
        col_idx = i % n
        sns.countplot(data=df, x=col, ax=axs[row, col_idx])
        tick_labels = df[col].unique()
        axs[row, col_idx].set_xticks(range(len(tick_labels)))
        axs[row, col_idx].set_xticklabels(tick_labels, rotation=90, fontsize=15)
        axs[row, col_idx].set_title(f'Distribution of {col}', fontsize=15)
        axs[row, col_idx].set_xlabel(col, fontsize=15)
        axs[row, col_idx].set_ylabel('Count', fontsize=15)

    for i in range(len(categorical_cols), m * n):
        row = i // n
        col_idx = i % n
        axs[row, col_idx].axis('off')

    plt.tight_layout()
    return fig

def save_plot_as_image(fig, image_path):
    fig.savefig(image_path, format='png')
    plt.close(fig) 

def detailed_eda(request):
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    df = pd.read_csv(file_path)
    columns = get_columns_for_dropdown()
    

    # Save plots
    numerical_image_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'images', 'numerical_plot.png')
    categorical_image_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'images', 'categorical_plot.png')
    print(numerical_image_path,categorical_image_path)
    context = {
        'numerical_plot': numerical_image_path,
        'categorical_plot': categorical_image_path,
        'df': df,
        'columns': columns,
    }
    return render(request, 'mlapp/detailed_eda.html', context)

def bivariate_analysis(df, target=None, n=3, plot_type='box'):
    import warnings
    if target is None:
        target = df.columns[-1]

    variables = df.drop(target, axis=1)
    numerical_cols = variables.select_dtypes(include=[np.number]).columns
    categorical_cols = variables.select_dtypes(include=['object', 'category']).columns

    if df[target].dtype in [np.int64, np.float64] and df[target].nunique() > 10:
        target_type = 'numerical'
    else:
        target_type = 'categorical'

    num_plots = len(numerical_cols) + len(categorical_cols)
    m = int(np.ceil(num_plots / n))
    fig, axes = plt.subplots(m, n, figsize=(5 * n, 6 * m))
    fig.suptitle(f'Bivariate Analysis: {target} vs Other Variables', y=1.02)

    warnings.filterwarnings('ignore')
    for idx, x_col in enumerate(variables):
        print(str(idx) + "/" + str(num_plots - 1), end="\r")
        ax = axes.flatten()[idx]

        if df[x_col].dtype in [np.int64, np.float64] and df[x_col].nunique() > 10:
            x_type = 'numerical'
        else:
            x_type = 'categorical'

        if target_type == 'numerical' and x_type == 'numerical':
            sns.scatterplot(x=x_col, y=target, data=df, ax=ax)
            ax.set_title(f'Scatter Plot: {x_col} vs {target}')

        elif target_type == 'categorical' and x_type == 'numerical':
            if plot_type == 'box':
                sns.boxplot(y=x_col, x=target, data=df, ax=ax)
                ax.set_title(f'Box Plot: {x_col} vs {target}')
            elif plot_type == 'swarm':
                sns.swarmplot(y=x_col, x=target, data=df, ax=ax, size=10)
                ax.set_title(f'Swarm Plot: {x_col} vs {target}')
                if len(ax.get_xticklabels()) >= 5:
                    label_types = [type(label.get_text()) for label in ax.get_xticklabels()]
                    if str in label_types:
                        ax.tick_params(axis='x', rotation=45)

        elif target_type == 'numerical' and x_type == 'categorical':
            if plot_type == 'box':
                sns.boxplot(x=target, y=x_col, data=df, ax=ax)
                ax.set_title(f'Box Plot: {x_col} vs {target}')
            elif plot_type == 'swarm':
                sns.swarmplot(x=target, y=x_col, data=df, ax=ax, size=10)
                ax.set_title(f'Swarm Plot: {x_col} vs {target}')
                if len(ax.get_xticklabels()) >= 5:
                    label_types = [type(label.get_text()) for label in ax.get_xticklabels()]
                    if str in label_types:
                        ax.tick_params(axis='x', rotation=45)
        elif target_type == 'categorical' and x_type == 'categorical':
            sns.countplot(x=x_col, hue=target, data=df, ax=ax)
            ax.set_title(f'Count Plot: {x_col} by {target}')
            if len(ax.get_xticklabels()) >= 5:
                label_types = [type(label.get_text()) for label in ax.get_xticklabels()]
                if str in label_types:
                    ax.tick_params(axis='x', rotation=45)

    for j in range(num_plots, m * n):
        axes.flatten()[j].axis('off')

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

@csrf_exempt
def generate_bivariate_plot(request):
    if request.method == 'GET':
        num_columns = int(request.GET.get('num_columns', 3))
        plot_type = request.GET.get('plot_type', 'box')
        target_column = request.GET.get('target_column', None)

        # Access the already uploaded DataFrame
        file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
        df = pd.read_csv(file_path) 

        plot_base64 = bivariate_analysis(df, target=target_column, n=num_columns, plot_type=plot_type)
        return JsonResponse({'plot_base64': plot_base64})



def plot_3var_combinations(df, variables=None, target=None, plot_types=None, n=4):
    default_plot_types = {
        'num3': ['3dscatter', 'pairplot'],
        'cat1_num2': ['box', 'grouped_scatter', 'violin', 'facet_grid'],
        'cat2_num1': ['grouped_box', 'violin', 'point', 'swarm'],
        'cat3': ['mosaic', '3dbar', 'heatmap']
    }
    
    
    if plot_types is None:
        plot_types = default_plot_types
    
    
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    num_cols = [col for col in num_cols if df[col].nunique() > 10]
    
    
    if variables is None:
        columns = df.columns.tolist()
        variables = [list(comb) for comb in itertools.combinations(columns, 3)]
    
    
    if target:
        variables = [comb for comb in variables if target in comb]
    
    
    m = int(np.ceil(len(variables) / n))
    fig, axes = plt.subplots(m, n, figsize=(5*n, 7*m))
    axes = axes.flatten()
    
    for i, (var1, var2, var3) in enumerate(variables):
        print(str(i+1)+"/"+str(len(variables)),end="\r")
        ax = axes[i]
        
        # Determine types of the variables
        types = [var1 in num_cols, var2 in num_cols, var3 in num_cols]
        
        if sum(types) == 3:  # num3
            plot_type = plot_types['num3'][0]  # Default to 3D scatter plot
            try:
                if plot_type == '3dscatter':
                    ax = fig.add_subplot(m, n, i+1, projection='3d')
                    ax.scatter(df[var1], df[var2], df[var3], c='b', marker='o')
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)
                    ax.set_zlabel(var3)
                elif plot_type == 'pairplot':
                    sns.pairplot(df[[var1, var2, var3]], ax=ax)
            except ValueError as e:
                ax.text(0.5, 0.5, str(e), fontsize=12, ha='center')
        
        elif sum(types) == 0:  # cat3
            plot_type = plot_types['cat3'][2]  # Default to mosaic plot
            try:
                if plot_type == 'mosaic':
                    mosaic(df, [var1, var2, var3], ax=ax)
                elif plot_type == 'heatmap':
                    contingency_table = pd.crosstab(df[var1], [df[var2], df[var3]])
                    sns.heatmap(contingency_table, cmap="YlGnBu", ax=ax)
                elif plot_type == '3dbar':
                    ax = fig.add_subplot(m, n, i+1, projection='3d')
                    x_pos = np.arange(len(df[var1].unique()))
                    y_pos = np.arange(len(df[var2].unique()))
                    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
                    x_pos = x_pos.flatten()
                    y_pos = y_pos.flatten()
                    z_pos = np.zeros_like(x_pos)
                    dx = dy = 0.8
                    dz = pd.crosstab(df[var1], [df[var2], df[var3]]).values.flatten()
                    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz)
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)
                    ax.set_zlabel(var3)
            except ValueError as e:
                ax.text(0.5, 0.5, str(e), fontsize=12, ha='center')
        
        elif sum(types) == 2:  # num vs num vs cat
            num_vars = [var1 if types[0] else None, var2 if types[1] else None, var3 if types[2] else None]
            num_vars = [v for v in num_vars if v is not None]
            cat_var = var1 if not types[0] else (var2 if not types[1] else var3)
            
            if df[num_vars[0]].max() >= df[num_vars[1]].max():
                y_var, x_var = num_vars
            else:
                x_var, y_var = num_vars
            
            sns.scatterplot(x=x_var, y=y_var, hue=cat_var, data=df, ax=ax)
        
        elif sum(types) == 1:  # cat1_num2
            num_var = [var1 if types[0] else (var2 if types[1] else var3)][0]
            cat_vars = [var for var in [var1, var2, var3] if var != num_var]
            
            if df[cat_vars[0]].nunique() >= df[cat_vars[1]].nunique():
                x_var = cat_vars[0]
                hue_var = cat_vars[1]
            else:
                x_var = cat_vars[1]
                hue_var = cat_vars[0]
            
            plot_type = plot_types['cat2_num1'][0]  # Default to grouped box plot
            try:
                if plot_type == 'grouped_box':
                    sns.boxplot(x=x_var, y=num_var, hue=hue_var, data=df, ax=ax)
                elif plot_type == 'violin':
                    sns.violinplot(x=x_var, y=num_var, hue=hue_var, data=df, ax=ax)
                elif plot_type == 'point':
                    sns.pointplot(x=x_var, y=num_var, hue=hue_var, data=df, ax=ax)
                elif plot_type == 'swarm':
                    sns.swarmplot(x=x_var, y=num_var, hue=hue_var, data=df, ax=ax)
            except ValueError as e:
                ax.text(0.5, 0.5, str(e), fontsize=12, ha='center')
        
        ax.set_title(f'{var1} vs {var2} vs {var3}')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

@csrf_exempt
def generate_multivariate_plot(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        variables = data.get('variables', []) 

        file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
        df = pd.read_csv(file_path) 

        plot_base64 = plot_3var_combinations(df, variables=variables)

        return JsonResponse({'plot_base64': plot_base64})

def encode_image_to_base64(fig):
    """Encodes a matplotlib figure to a base64 string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64

def get_columns_for_dropdown():
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    df = pd.read_csv(file_path)
    df_dummies = pd.get_dummies(df, drop_first=True)
    return df_dummies.columns.tolist()



def target_correlation_Analysis(df, target):
    # Convert categorical variables to dummy variables
    df_dummies = pd.get_dummies(df, drop_first=True)
    
    # Ensure the target is a column in the DataFrame
    if target not in df_dummies.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    
    features = df_dummies.drop(target, axis=1)
    correlations = features.corrwith(df_dummies[target])
    
    correlation_df = pd.DataFrame({'Feature': correlations.index, 'Correlation': correlations.values})
    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)
    
    plt.figure(figsize=(12, 14))  # Increase figure size for better readability
    sns.barplot(x='Correlation', y='Feature', data=correlation_df, hue='Feature', palette='viridis', legend=False)
    
    plt.title(f'Correlation of Features with {target}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    
    # Rotate y-axis labels if they are too long
    plt.yticks(rotation=0, fontsize=10)  # Rotate if needed and adjust fontsize
    
    # Ensure layout is tight to prevent clipping of labels
    plt.tight_layout()
    
    # Save plot to a BytesIO object and encode as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  # Close the plot to avoid resource warnings
    
    return base64.b64encode(image_png).decode('utf-8')

@csrf_exempt
def generate_correlation_plot(request):
    if request.method == 'GET':
        target_column = request.GET.get('target_column', None)

        # Access the already uploaded DataFrame
        file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
        df = pd.read_csv(file_path)

        plot_base64 = target_correlation_Analysis(df, target=target_column)
        return JsonResponse({'plot_base64': plot_base64})


#advanced_analysis stuff
def advanced_analysis(request):
    return data_statistics(request)

def data_statistics(request):
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    
    rows_to_display = request.GET.get('rows_to_display', '5')
    
    data_head, data_info, missing_values, summary_statistics = extract_data_info(file_path, rows_to_display)
    
    context = {
        'data_head': data_head,
        'data_info': data_info,
        'missing_values': missing_values,
        'summary_statistics': summary_statistics,
        'rows_to_display': rows_to_display
    }
    
    return render(request, 'mlapp/advanced_analysis.html', context)

def extract_data_info(file_path, rows_to_display=5):
    df = pd.read_csv(file_path)
    
    # Extract data head
    if rows_to_display == 'all':
        data_head = df.to_html(classes='table table-striped table-bordered')
    else:
        data_head = df.head(int(rows_to_display)).to_html(classes='table table-striped table-bordered')

    # Extract data info
    buffer = io.StringIO()
    df.info(memory_usage=False, buf=buffer)
    info_output = buffer.getvalue().split("\n")
    info_lines = info_output[1:]
    data_info = "\n".join(info_lines)

    # Extract missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_values = missing_values.to_html(classes='table table-striped table-bordered') if not missing_values.empty else "<p>No missing values found.</p>"

    # Extract summary statistics for numerical variables
    summary_statistics = df.describe().to_html(classes='table table-striped table-bordered')

    return data_head, data_info, missing_values, summary_statistics

#data cleaning part

def handle_missing_values(df, method="remove", k=5):
    """Handles missing values based on the specified method."""
    print("Missing values before handling:", df.isnull().sum()[df.isnull().sum() > 0])

    if method == "KNN":
        imputer = KNNImputer(n_neighbors=k)
        imputed_data = imputer.fit_transform(df)
        df = pd.DataFrame(imputed_data, columns=df.columns)

    elif method == "remove":
        df.dropna(inplace=True)

    elif method in ("mean", "median"):
        imputation_func = df.mean if method == "mean" else df.median
        df.fillna(imputation_func(), inplace=True)

    elif method == "mode":
        df.fillna(df.mode().iloc[0], inplace=True)

    else:
        raise ValueError(f"Unsupported method: {method}")

    print("Missing values after handling:", df.isnull().sum()[df.isnull().sum() > 0])
    return df

def generate_box_plot(df, n=4):
    """Generates a box plot and returns it as a base64 encoded image."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    m = int(np.ceil(len(numerical_cols) / n))

    fig, axes = plt.subplots(m, n, figsize=(16, m * n))
    fig.suptitle('Boxplot of Numerical Variables', fontsize=16, y=1.0)

    for i, col in enumerate(numerical_cols):
        row = i // n
        col_idx = i % n
        ax = axes[row, col_idx]
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col, fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')

    for i in range(len(numerical_cols), m * n):
        row = i // n
        col_idx = i % n
        axes[row, col_idx].axis('off')

    plt.tight_layout()
    return encode_image_to_base64(fig)

def remove_outliers(df, method='iqr', z=3, uniques=6):
    """Removes outliers from numerical columns."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    modified_df = df.copy()

    for column in numerical_cols:
        if modified_df[column].nunique() > uniques:
            if method == 'iqr':
                Q1 = modified_df[column].quantile(0.25)
                Q3 = modified_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                modified_df = modified_df[(modified_df[column] >= lower_bound) & (modified_df[column] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((modified_df[column] - modified_df[column].mean()) / modified_df[column].std())
                modified_df = modified_df[z_scores <= z]
    return modified_df

def suggest_col_to_remove(df):
    """Suggests columns for removal and returns their plots."""
    should_remove = []
    plot_data = {}

    for col in df.columns:
        if df[col].nunique() <= 1 or df[col].nunique() == len(df[col]):
            should_remove.append(col)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            sns.scatterplot(x=df.index, y=df[col], ax=ax1)
            sns.histplot(df[col], ax=ax2)
            ax1.set_title(col)
            ax2.set_title(col)
            plt.tight_layout()
            plot_data[col] = encode_image_to_base64(fig)

    return should_remove, plot_data

def encode_image_to_base64(fig):
    """Encodes a matplotlib figure to a base64 string."""
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

@csrf_exempt
def clean_data(request):
    """Handles all data cleaning actions."""
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'cleaned_dataset.csv')
    original_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    df = pd.read_csv(file_path)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            action = data.get('action')

            if action == 'suggest_columns':
                suggested_columns, plot_data = suggest_col_to_remove(df)
                return JsonResponse({'columns': suggested_columns, 'plot_data': plot_data})

            elif action == 'remove_columns':
                columns_to_remove = data.get('columns')
                if isinstance(columns_to_remove, list):  # Ensure it's a list
                    print(columns_to_remove)
                    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
                    df.to_csv(file_path, index=False)
                    return JsonResponse({"message": "Columns removed successfully"})
                else:
                    return JsonResponse({'error': 'Invalid columns data. Must be a list.'}, status=400)

            elif action == 'handle_missing_values':
                method = data.get('method') 
                k = int(data.get('k', 5))  
                df = handle_missing_values(df, method=method, k=k)
                df.to_csv(file_path, index=False)
                return JsonResponse({'message': 'Missing values handled successfully'})

            elif action == 'remove_outliers':
                method = data.get('method')
                z_threshold = float(data.get('z_threshold', 3))  # Default z-score threshold is 3
                df = remove_outliers(df, method=method, z=z_threshold)
                df.to_csv(file_path, index=False)
                return JsonResponse({'message': 'Outliers removed successfully'})

            elif action == 'generate_box_plot':
                box_plot_base64 = generate_box_plot(df)
                return JsonResponse({'box_plot': box_plot_base64})

            elif action == 'replace_file':
                df.to_csv(original_file_path, index=False)
                # Generate and save plots
                numerical_fig = plot_numerical_cols(df)
                categorical_fig = plot_categorical_cols(df)

                # Ensure the images directory exists
                images_dir = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'images')
                os.makedirs(images_dir, exist_ok=True)

                numerical_image_path = os.path.join(images_dir, 'numerical_plot.png')
                categorical_image_path = os.path.join(images_dir, 'categorical_plot.png')
                
                save_plot_as_image(numerical_fig, numerical_image_path)
                save_plot_as_image(categorical_fig, categorical_image_path)
                return JsonResponse({'message': 'File replaced successfully'})

            elif action == 'refresh_preview':
                df_preview = df.head().to_html(classes='table table-striped')
                return JsonResponse({'df': df_preview})

            else:
                return JsonResponse({'error': 'Invalid action.'}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

    elif request.method == 'GET':
        return render(request, 'mlapp/clean_data.html', {'df': df.head().to_html(classes='table table-striped')})

    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)

#Predictions:
# Load the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
model1 = joblib.load(os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'attrition.pkl'))

         
def predict_attrition(request):
    if request.method == 'POST':
        # Extract data from the form
        data = {
            'Age': int(request.POST['age']),
            'Education': int(request.POST['education']),
            'OverTime_Yes': int(request.POST['overtime']),
            'DistanceFromHome': int(request.POST['distance']),
            'EnvironmentSatisfaction': int(request.POST['environment']),
            'JobInvolvement': int(request.POST['involvement']),
            'JobRole_Research Scientist': int(request.POST['jobrole_research']),
            'JobSatisfaction': int(request.POST['jobsatisfaction']),
            'Gender_Male': int(request.POST['gender']),
            'BusinessTravel_Travel_Frequently': int(request.POST['businesstravel']),
            'MonthlyIncome': int(request.POST['income']),
            'NumCompaniesWorked': int(request.POST['numcompanies']),
            'StockOptionLevel': int(request.POST['stockoption']),
            'PerformanceRating': int(request.POST['performance']),
            'RelationshipSatisfaction': int(request.POST['relationship']),
            'YearsAtCompany': int(request.POST['yearsatcompany']),
            'TotalWorkingYears': int(request.POST['totalworkingyears']),
            'JobLevel': int(request.POST['joblevel']),
            'WorkLifeBalance': int(request.POST['worklifebalance']),
            'YearsSinceLastPromotion': int(request.POST['yearssincelastpromotion']),
            'YearsWithCurrManager': int(request.POST['yearswithcurrmanager']),
            'JobRole_Sales Executive': int(request.POST['jobrole_sales']),
            'MaritalStatus_Single': int(request.POST['maritalstatus'])
        }

        input_data = pd.DataFrame([data])

        # Apply one-hot encoding
        # input_data = pd.get_dummies(input_data)

        prediction_proba = model1.predict_proba(input_data)[0][1]
        prediction = model1.predict(input_data)[0]
        print(prediction)
        # Convert the output to native Python types
        prediction_proba = float(prediction_proba)
        return JsonResponse({
            'prediction': bool(model1.predict(input_data)[0]),
            'probability': prediction_proba * 100
        })

    return render(request, 'mlapp/attrition_prediction.html')


def process_csv(request):
    # Load the CSV file
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    df = pd.read_csv(file_path)
    
    # Save the original columns to re-add later
    original_columns = df.columns.tolist()
    
    # Assuming preprocessing (e.g., one-hot encoding) was done in the pipeline, get the expected feature names
    if hasattr(model1, 'named_steps') and 'preprocessor' in model1.named_steps:
        preprocessor = model1.named_steps['preprocessor']
        expected_columns = preprocessor.transformers_[1][1].get_feature_names_out()
    else:
        # Fallback: use the columns from the training data (if available)
        expected_columns = model1.feature_names_in_
    
    # Preprocessing: One-hot encode categorical variables to match training data
    categorical_features = ['BusinessTravel', 'Gender', 'JobRole', 'MaritalStatus']
    df_preprocessed = pd.get_dummies(df, columns=categorical_features)
    
    # Handle any missing columns after one-hot encoding (fill with 0s)
    for col in expected_columns:
        if col not in df_preprocessed.columns:
            df_preprocessed[col] = 0

    # Ensure column order matches
    df_preprocessed = df_preprocessed.reindex(columns=expected_columns, fill_value=0)
    
    # Make predictions
    predictions = model1.predict(df_preprocessed)
    prediction_proba = model1.predict_proba(df_preprocessed)[:, 1] * 100  # Get percentage probability of attrition

    # Round the probabilities to 3 decimal places
    prediction_proba = prediction_proba.round(3)

    # Add the predictions to the original DataFrame
    df['Attrition_Prediction'] = predictions
    df['Attrition_Probability'] = prediction_proba
    
    # Save the updated file
    output_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'attrition_prediction.csv')
    df.to_csv(output_file_path, index=False)
    
    # Provide the file as a download
    with open(output_file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="attrition_prediction.csv"'
    
    return response

# Load the trained model
model2 = joblib.load(os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'PR.pkl'))

def performance_prediction(request):
    if request.method == 'POST':
        try:
            print("Received POST request")
            # Extract data from the form
            data = {
                'DailyRate': int(request.POST.get('DailyRate', 0)),
                'HourlyRate': int(request.POST.get('HourlyRate', 0)),
                'DistanceFromHome': int(request.POST.get('distance', 0)),
                'PercentSalaryHike': int(request.POST.get('percentsalaryhike', 0)),
                'TotalWorkingYears': int(request.POST.get('totalworkingyears', 0)),
                'YearsAtCompany': int(request.POST.get('yearsatcompany', 0)),
            }

            # Define the correct column order
            expected_columns = ['DailyRate', 'DistanceFromHome', 'HourlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany']
            input_data = pd.DataFrame([data], columns=expected_columns)

            # Ensure the DataFrame columns are in the correct order
            input_data = input_data[expected_columns]
            print("Reordered Columns in DataFrame:", input_data.columns.tolist())

            # Make prediction
            prediction = model2.predict(input_data)[0]
            prediction_proba = model2.predict_proba(input_data).max() * 100  # Get highest probability percentage
            print("PR=",prediction)
            # Convert the output to native Python types
            prediction = int(prediction)
            prediction_proba = float(prediction_proba)
            print("PR=",prediction)
            return JsonResponse({
                'prediction': prediction,
                'probability': prediction_proba
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    # Render the prediction form for GET requests
    return render(request, 'mlapp/performance_prediction.html')






def process_csv2(request):
    # Load the CSV file
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    df = pd.read_csv(file_path)
    
    # Define the columns that the model expects
    required_columns = ['DailyRate', 'DistanceFromHome', 'HourlyRate', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany']
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the input data: {', '.join(missing_columns)}")
    
    # Add missing columns with default values if necessary
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default values

    # Reorder columns to match the model's expected feature order
    input_data = df[required_columns]

    # Print out the reordered DataFrame columns for debugging
    print("Reordered Columns in DataFrame:", input_data.columns.tolist())
    
    # Convert DataFrame to numpy array to ensure compatibility
    input_data = input_data.values

    # Make predictions
    try:
        predictions = model2.predict(input_data)

        # Add the predictions to the original DataFrame
        df['PerformanceRating_Prediction'] = predictions

        # Save the updated file with all original columns plus predictions
        output_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'performance_prediction.csv')
        df.to_csv(output_file_path, index=False)
        
        # Provide the file as a download
        with open(output_file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="performance_prediction.csv"'
        
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return HttpResponse(f"An error occurred: {e}", status=500)

#Loading models for SHP
def load_models():
    models = []
    for fold in range(1, 6):  # Adjust the range according to your model count
        model_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', f'SHP2{fold}.pkl')
        if os.path.exists(model_path):
            models.append(joblib.load(model_path))
        else:
            print(f"Model file {model_path} not found.")
    return models

def predict_salary_hike(request):
    if request.method == 'POST':
        try:
            # Extract data from the form
            jobinvolvement = int(request.POST['jobinvolvement'])
            performancerating = int(request.POST['performancerating'])
            trainingtimeslastyear = int(request.POST['trainingtimeslastyear'])
            yearsatthecompany = int(request.POST['yearsatthecompany'])
            yearsincurrentrole = int(request.POST['yearsincurrentrole'])
            yearssincelastpromotion = int(request.POST['yearssincelastpromotion'])

            businesstravel = request.POST['businesstravel']
            department = request.POST['department']
            gender = request.POST['gender']
            jobrole = request.POST['jobrole']

            # Create a data dictionary with expected columns
            data = {
                'JobInvolvement': jobinvolvement,
                'PerformanceRating': performancerating,
                'TrainingTimesLastYear': trainingtimeslastyear,
                'YearsAtCompany': yearsatthecompany,
                'YearsInCurrentRole': yearsincurrentrole,
                'YearsSinceLastPromotion': yearssincelastpromotion,
                # Initialize binary columns
                'BusinessTravel_Travel_Frequently': 0,
                'BusinessTravel_Travel_Rarely': 0,
                'Department_Research & Development': 0,
                'Gender_Male': 0,
                'JobRole_Human Resources': 0,
                'JobRole_Manager': 0,
                'JobRole_Research Scientist': 0,
                'JobRole_Sales Representative': 0
            }

            # Map dropdown values to binary columns
            if businesstravel == 'Travel_Frequently':
                data['BusinessTravel_Travel_Frequently'] = 1
            elif businesstravel == 'Travel_Rarely':
                data['BusinessTravel_Travel_Rarely'] = 1

            if department == 'R&D':
                data['Department_Research & Development'] = 1

            if gender == 'Male':
                data['Gender_Male'] = 1

            if jobrole == 'Human Resources':
                data['JobRole_Human Resources'] = 1
            elif jobrole == 'Manager':
                data['JobRole_Manager'] = 1
            elif jobrole == 'Research Scientist':
                data['JobRole_Research Scientist'] = 1
            elif jobrole == 'Sales Representative':
                data['JobRole_Sales Representative'] = 1

            # Convert the dictionary to a DataFrame
            input_data = pd.DataFrame([data])

            # Load all models
            models = load_models()

            # Aggregate predictions from all models
            predictions = [model.predict(input_data)[0] for model in models]
            final_prediction = np.mean(predictions)

            # Return the result as a JSON response
            return JsonResponse({
                'prediction': float(final_prediction),
            })

        except KeyError as e:
            # Handle missing keys
            return JsonResponse({'error': f'Missing form field: {e}'}, status=400)
        except ValueError as e:
            # Handle value conversion errors
            return JsonResponse({'error': f'Invalid value: {e}'}, status=400)
    
    return render(request, 'mlapp/salary_prediction.html')

def process_csv3(request):
    # Load the CSV file
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
    df = pd.read_csv(file_path)
    
    # Define the expected feature columns in the correct order
    expected_columns = [
        'JobInvolvement', 'PerformanceRating', 'TrainingTimesLastYear', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'BusinessTravel_Travel_Frequently',
        'BusinessTravel_Travel_Rarely', 'Department_Research & Development', 'Gender_Male',
        'JobRole_Human Resources', 'JobRole_Manager', 'JobRole_Research Scientist', 'JobRole_Sales Representative'
    ]

    # Prepare the data: Ensure the CSV data matches the expected columns
    df_preprocessed = df.copy()
    
    # Assuming preprocessing is done to create these columns
    # Add any necessary dummy variables if missing
    categorical_features = {
        'BusinessTravel': ['BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely'],
        'Department': ['Department_Research & Development'],
        'Gender': ['Gender_Male'],
        'JobRole': ['JobRole_Human Resources', 'JobRole_Manager', 'JobRole_Research Scientist', 'JobRole_Sales Representative']
    }
    
    # One-hot encode categorical variables
    df_preprocessed = pd.get_dummies(df_preprocessed, columns=categorical_features.keys())
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df_preprocessed.columns:
            df_preprocessed[col] = 0

    # Reorder columns to match model input
    df_preprocessed = df_preprocessed[expected_columns]

    # Load all models
    models = load_models()

    # Make predictions using all models
    all_predictions = np.array([model.predict(df_preprocessed) for model in models])
    
    # Average predictions from all models
    mean_predictions = np.mean(all_predictions, axis=0)
    
    # Convert predictions to integer values
    predictions_int = mean_predictions.round().astype(int)
    
    # Add the predictions to the original DataFrame
    df['Predicted_Salary_Hike'] = predictions_int
    
    # Save the updated file
    output_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'salary_prediction.csv')
    df.to_csv(output_file_path, index=False)
    
    # Provide the file as a download
    with open(output_file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="salary_prediction.csv"'
    
    return response

def update_profile(request):
    return render(request, 'mlapp/update_profile.html')

def salary_hike_recommendation(request):
    return render(request, 'mlapp/salary_hike_recommendation.html')

def predictions(request):
    return render(request, 'mlapp/predictions.html')

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assuming FAISS retriever is already set up
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import pickle
from typing import List, Tuple
from langchain.schema import Document

# Initialize Ollama LLM
ollama_llm = Ollama(model='llama3.1:latest')

base_dir = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp')

# Load FAISS index from static/mlapp
index_file_path = os.path.join(base_dir, 'vector_storage.index')
index = faiss.read_index(index_file_path)

# Load docstore and index_to_docstore_id from static/mlapp
docstore_file_path = os.path.join(base_dir, 'docstore.pkl')
index_to_docstore_id_file_path = os.path.join(base_dir, 'index_to_docstore_id.pkl')

with open(docstore_file_path, 'rb') as f:
    docstore = pickle.load(f)

with open(index_to_docstore_id_file_path, 'rb') as f:
    index_to_docstore_id = pickle.load(f)

# Load embeddings model
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
embedding_function = embeddings.embed_documents

# Modify the similarity_search_with_score_by_vector method in the FAISS class
class FAISS(FAISS):  # Inherit from the original FAISS class
    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, filter=None, fetch_k=20, **kwargs
    ) -> List[Tuple[Document, float]]:
        if self._normalize_L2:
            faiss.normalize_L2(embedding)
        # Handle the case where index.search returns more than two values
        # Convert the embedding list to a NumPy array
        embedding_array = np.array(embedding, dtype='float32')  # Convert to NumPy array
        search_results = self.index.search(embedding_array, k if filter is None else fetch_k)  # Use the array
        if len(search_results) == 2:
            scores, indices = search_results 
        else: 
            scores, indices, *other_values = search_results  # Unpack additional values if present
            print("Warning: FAISS index returned additional values:", other_values)  # Log a warning
        # Handle the case when no results are found
        if len(indices[0]) == 0:
            return []  # Return an empty list if no results
        else:
            docs_and_scores = [
                (self.docstore.search(self.index_to_docstore_id[idx]), score)
                for score, idx in zip(scores[0], indices[0])
            ]
            return docs_and_scores

# Recreate FAISS vector storage using the modified class
vector_storage = FAISS(
    index=index, 
    docstore=docstore, 
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_function
)

# Get retriever
retriever = vector_storage.as_retriever()
# Define the prompt template for filtering
filter_template = """
You are a Python data analysis assistant. Your task is to generate a single line of Python code to filter a DataFrame named 'df' based on the user's question. The DataFrame contains employee data with the following columns:

Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

The dataset contains the following key features:

Age: The age of the employee.
Attrition: Indicates whether the employee has left the company (Yes/No).
Business Travel: Frequency of travel for business purposes (Non-Travel, Travel_Rarely, Travel_Frequently).
Daily Rate: The daily rate of pay for the employee.
Department: The department where the employee works (Sales, Research & Development, Human Resources).
Distance from Home: The distance between the employee's home and workplace.
Education: The education level of the employee (1 to 5).
Education Field: The field in which the employee received their education (Life Sciences, Medical, Marketing, Technical Degree, etc.).
Environment Satisfaction: Employee's satisfaction with their work environment (1 to 4).
Gender: Gender of the employee (Male/Female).
Hourly Rate: The hourly rate of pay for the employee.
Job Involvement: Employee's involvement in their job (1 to 4).
Job Level: The job level of the employee (1 to 5).
Job Role: The role of the employee within the organization (Sales Executive, Research Scientist, Laboratory Technician, etc.).
Job Satisfaction: Employee's satisfaction with their job (1 to 4).
Marital Status: The marital status of the employee (Single, Married, Divorced).
Monthly Income: The monthly income of the employee.
Monthly Rate: The monthly rate of pay for the employee.
Num Companies Worked: The number of companies the employee has worked for.
Overtime: Whether the employee works overtime (Yes/No).
Percent Salary Hike: The percentage increase in salary over the last year.
Performance Rating: The performance rating of the employee (1 to 4).
Relationship Satisfaction: Employee's satisfaction with their relationships at work (1 to 4).
Stock Option Level: The stock option level of the employee (0 to 3).
Total Working Years: The total number of years the employee has worked.
Training Times Last Year: The number of times the employee underwent training in the last year.
Work Life Balance: Employee's work-life balance rating (1 to 4).
Years at Company: The number of years the employee has worked at the current company.
Years in Current Role: The number of years the employee has been in their current role.
Years Since Last Promotion: The number of years since the employee's last promotion.
Years with Current Manager: The number of years the employee has worked with their current manager.

Key points to remember:
1. Return ONLY the Python code, without any explanations or additional text.
2. The code should be a single line that can be directly executed.
3. Use proper Python syntax and DataFrame operations.
4. Ensure the code is efficient and follows best practices for pandas operations.
5. If multiple conditions are needed, use bitwise operators (&, |) instead of 'and', 'or' and don't forget to include the conditions in the brackets like (condition1) & (condition2), i repeat, don't forget to include the conditions in the brackets this is important.
6. Handle potential errors, such as non-existent columns or invalid comparisons.

Examples:

Question: Employees with salary greater than 10k?
Answer: df[df['MonthlyIncome'] > 10000]

Question: Female employees in the Sales department?
Answer: df[(df['Gender'] == 'Female') & (df['Department'] == 'Sales')]

Question: Employees with high job satisfaction and low attrition?
Answer: df[(df['JobSatisfaction'] >= 3) & (df['Attrition'] == 'No')]

Context: {context}
Question: {question}

Remember, provide only the code, nothing else.
"""


graph_template = """
You are a Python data visualization expert. Your task is to generate Python code to create a meaningful graph based on the user's question using a DataFrame named 'df'. The DataFrame contains employee data with the following columns:

Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
The dataset contains the following features:

Age: The age of the employee.
Attrition: Indicates whether the employee has left the company (Yes/No).
Business Travel: Frequency of travel for business purposes (Non-Travel, Travel_Rarely, Travel_Frequently).
Daily Rate: The daily rate of pay for the employee.
Department: The department where the employee works (Sales, Research & Development, Human Resources).
Distance from Home: The distance between the employee's home and workplace.
Education: The education level of the employee (1 to 5).
Education Field: The field in which the employee received their education (Life Sciences, Medical, Marketing, Technical Degree, etc.).
Environment Satisfaction: Employee's satisfaction with their work environment (1 to 4).
Gender: Gender of the employee (Male/Female).
Hourly Rate: The hourly rate of pay for the employee.
Job Involvement: Employee's involvement in their job (1 to 4).
Job Level: The job level of the employee (1 to 5).
Job Role: The role of the employee within the organization (Sales Executive, Research Scientist, Laboratory Technician, etc.).
Job Satisfaction: Employee's satisfaction with their job (1 to 4).
Marital Status: The marital status of the employee (Single, Married, Divorced).
Monthly Income: The monthly income of the employee.
Monthly Rate: The monthly rate of pay for the employee.
Num Companies Worked: The number of companies the employee has worked for.
Overtime: Whether the employee works overtime (Yes/No).
Percent Salary Hike: The percentage increase in salary over the last year.
Performance Rating: The performance rating of the employee (1 to 4).
Relationship Satisfaction: Employee's satisfaction with their relationships at work (1 to 4).
Stock Option Level: The stock option level of the employee (0 to 3).
Total Working Years: The total number of years the employee has worked.
Training Times Last Year: The number of times the employee underwent training in the last year.
Work Life Balance: Employee's work-life balance rating (1 to 4).
Years at Company: The number of years the employee has worked at the current company.
Years in Current Role: The number of years the employee has been in their current role.
Years Since Last Promotion: The number of years since the employee's last promotion.
Years with Current Manager: The number of years the employee has worked with their current manager.


Key points to remember:
1. Return ONLY the Python code, without any explanations or additional text.
2. Use matplotlib.pyplot as plt for plotting.
3. Always include proper labels for axes and a title for the graph.
4. Choose an appropriate graph type based on the data and question (e.g., bar plot, histogram, scatter plot, box plot).
5. If filtering is required, include the filtering code before plotting.
6. If multiple conditions are needed for filtering, use bitwise operators (&, |) instead of 'and', 'or' and don't forget to include the conditions in the brackets like (condition1) & (condition2).
7. For categorical data, consider using bar plots or pie charts.
8. For numerical data, consider using histograms, scatter plots, or box plots.
9. Use color to enhance the readability of the graph when appropriate.
10. Include a single plt.show() at the end of your code.
11. Assume all necessary libraries are already imported.

Examples:

Question: Distribution of employee ages?
Answer:
plt.hist(df['Age'], bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Employee Ages')
plt.show()

Question: Job satisfaction across departments?
Answer:
dept_satisfaction = df.groupby('Department')['JobSatisfaction'].mean()
plt.bar(dept_satisfaction.index, dept_satisfaction.values)
plt.xlabel('Department')
plt.ylabel('Average Job Satisfaction')
plt.title('Job Satisfaction Across Departments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

Question: Relationship between years at company and monthly income?
Answer:
plt.scatter(df['YearsAtCompany'], df['MonthlyIncome'])
plt.xlabel('Years at Company')
plt.ylabel('Monthly Income')
plt.title('Years at Company vs Monthly Income')
plt.show()

Context: {context}
Question: {question}

Remember, provide only the code, nothing else.
"""

@csrf_exempt
def ai_filter(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            data = json.loads(request.body)
            user_input = data.get('question', '')

            # Generate the filtering code
            result = generate_ai_filter_code(user_input)
            return JsonResponse({'result': result})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def ai_graph(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            data = json.loads(request.body)
            user_input = data.get('question', '')

            # Generate the graph plotting code
            result = generate_ai_graph_code(user_input)
            return JsonResponse({'result': result})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def run_code(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        code = data.get('code')
        
        # Set up the environment for code execution
        file_path = os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv')
        df = pd.read_csv(file_path)
        
        # Capture output
        output = io.StringIO()
        result_type = 'text'
        result_data = None
        
        try:
            # Create a local namespace for execution
            local_vars = {'df': df, 'plt': plt}
            
            # Execute the code
            if '\n' in code:
                # Multi-line code
                exec(code, globals(), local_vars)
                if plt.get_fignums():
                    img = io.BytesIO()
                    plt.savefig(img, format='png')
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode()
                    output.write(f'<img src="data:image/png;base64,{plot_url}">')
                    plt.close()
                    result_type = 'image'
                    result_data = plot_url
            else:
                # Single-line code
                result = eval(code, globals(), local_vars)
                if isinstance(result, pd.DataFrame):
                    html_table = result.to_html(classes=['table', 'table-striped', 'table-bordered', 'table-hover', 'table-responsive'])
                    output.write(html_table)
                    result_type = 'table'
                    result_data = result.to_csv(index=False)
                else:
                    output.write(str(result))
            
            # Check for any created DataFrames in local_vars
            for var_name, var_value in local_vars.items():
                if var_name != 'df' and isinstance(var_value, pd.DataFrame):
                    html_table = var_value.head(50).to_html(classes=['table', 'table-striped', 'table-bordered', 'table-hover', 'table-responsive'])
                    output.write(f"<h3>{var_name}:</h3>")
                    output.write(html_table)
                    result_type = 'table'
                    result_data = var_value.to_csv(index=False)
            
            if not output.getvalue():
                output.write("Code executed successfully, but no output was generated.")
            
        except Exception as e:
            output.write(str(e))
        
        return JsonResponse({
            'output': output.getvalue(),
            'result_type': result_type,
            'result_data': result_data
        })
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def download_result(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        result_type = data.get('result_type')
        result_data = data.get('result_data')
        
        if result_type == 'table':
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="filtered_data.csv"'
            response.write(result_data)
        elif result_type == 'image':
            response = HttpResponse(content_type='image/png')
            response['Content-Disposition'] = 'attachment; filename="plot.png"'
            response.write(base64.b64decode(result_data))
        else:
            return JsonResponse({'error': 'Invalid result type'})
        
        return response
    
    return JsonResponse({'error': 'Invalid request method'})




# @csrf_exempt
# def run_code(request):
#     df=pd.read_csv(os.path.join(settings.STATICFILES_DIRS[0], 'mlapp', 'file.csv'))
#     if request.method == 'POST':
#         try:
#             # Get user code from request
#             user_code = json.loads(request.body).get('output', '')

#             # Create a context for exec to run user code
#             context = {'df': df, 'plt': plt}
            
#             # Execute the user code within the context
#             exec(user_code, globals(), context)

#             # Extract filtered DataFrame if it exists in context
#             filtered_df = context.get('df', df)

#             # Convert DataFrame to JSON
#             data = filtered_df.to_dict(orient='records')
            
#             # Get plot image if plt is used
#             plot_image = None
#             if 'plt' in context and hasattr(context['plt'], 'savefig'):
#                 buf = BytesIO()
#                 context['plt'].savefig(buf, format='png')
#                 buf.seek(0)
#                 plot_image = base64.b64encode(buf.read()).decode('utf-8')
#                 buf.close()
#                 plt.close()  # Close the figure to avoid excessive memory usage

#             result = {
#                 'message': 'Code executed successfully!',
#                 'data': data,
#                 'plot': plot_image
#             }
#         except json.JSONDecodeError:
#             result = {
#                 'message': 'Invalid JSON',
#                 'data': [],
#                 'plot': None
#             }
#         except Exception as e:
#             result = {
#                 'message': str(e),
#                 'data': [],
#                 'plot': None
#             }

#         return JsonResponse(result)
#     else:
#         return JsonResponse({'error': 'Invalid request method'}, status=405)
    
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# Chain components together (assuming these are already defined elsewhere in your code)
filter_prompt = PromptTemplate.from_template(template=filter_template)
filter_chain = RunnableParallel(context=retriever, question=RunnablePassthrough()) | filter_prompt | ollama_llm | StrOutputParser()

graph_prompt = PromptTemplate.from_template(template=graph_template)
graph_chain = RunnableParallel(context=retriever, question=RunnablePassthrough()) | graph_prompt | ollama_llm | StrOutputParser()

# # Define the prompt templates
# filter_prompt = PromptTemplate.from_template(template=filter_template)
# graph_prompt = PromptTemplate.from_template(template=graph_template)

# # Chain components together
# filter_chain = RunnableParallel(context=retriever, question=RunnablePassthrough()) | filter_prompt | ollama_llm | StrOutputParser()
# graph_chain = RunnableParallel(context=retriever, question=RunnablePassthrough()) | graph_prompt | ollama_llm | StrOutputParser()

# ... (previous code remains the same)

def generate_ai_graph_code(user_input):
    try:
        result = graph_chain.invoke({"context": retriever, "question": user_input})
        return str(result)  # Convert the result to a string, regardless of its type
    except Exception as e:
        return f"Error generating graph code: {str(e)}"

def generate_ai_filter_code(user_input):
    try:
        result = filter_chain.invoke({"context": retriever, "question": user_input})
        return str(result)  # Convert the result to a string, regardless of its type
    except Exception as e:
        return f"Error generating filter code: {str(e)}"

def ai(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        action = request.POST.get('action')
        
        if action == 'filter':
            ai_response = generate_ai_filter_code(user_input)
        elif action == 'plot':
            ai_response = generate_ai_graph_code(user_input)
        else:
            ai_response = "Invalid action"
        
        return render(request, 'mlapp/ai.html', {'ai_response': ai_response})
    
    return render(request, 'mlapp/ai.html')

# ... (rest of the code remains the same)

# # Assuming you're using a simple pass-through, you can directly pass the values
# def generate_ai_graph_code(user_input):
#     # Assuming you're using the graph_chain directly
#     result = graph_chain.invoke({"context": retriever, "question": user_input})
#     return result["output"]

# def generate_ai_filter_code(user_input):
#     # Assuming you're using the filter_chain directly
#     result = filter_chain.invoke({"context": retriever, "question": user_input})
#     return result["output"]