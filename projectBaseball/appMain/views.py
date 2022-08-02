from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

def choose(request):
    return render(request, 'appMain/choose.html')