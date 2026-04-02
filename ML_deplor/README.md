# AgriOpti ML Deployment

This directory contains everything you need to deploy your ML API to Render.com. We have prepared this folder specifically so that it is isolated from your React frontend and easy to host as a standalone Web Service.

## Step 1: Upload to GitHub

1. Open your terminal or Git bash and navigate to this folder:
   ```bash
   cd c:/Users/kcham/Downloads/Full_Working/AgriOpti/ML_Deploy
   ```
2. Initialize a Git repository inside this folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for ML API"
   ```
3. Go to [GitHub](https://github.com/) and create a new **public** repository (e.g., `AgriOpti-ML`). Do not initialize it with a README or .gitignore.
4. Follow the instructions on GitHub to push your local repository. It will look like this:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/AgriOpti-ML.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy to Render.com

1. Go to your [Render Dashboard](https://dashboard.render.com/) and click **New+** > **Web Service**.
2. Connect your GitHub account and select your newly created repository (`AgriOpti-ML`).
3. Fill in the deployment details:
   - **Name**: `agriopti-ml-api` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Region**: (Choose closest to you)
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app` (This is also defined in the `Procfile`)
4. **Important Environment Variables**:
   Scroll down to the **Advanced** section and click **Add Environment Variable**:
   - Key: `PYTHON_VERSION`
   - Value: `3.10.0` (Render defaults to Python 3.7.10 if not set, which will cause deployment to fail since `google-genai` and TensorFlow require newer Python).
5. Choose the **Free** instance type (or a paid one for better speed, given the large TensorFlow models involved) and click **Create Web Service**.

> **Note on Free tier**: The ML models (`app.py`, `tensorflow`) are quite large and might take some time to start up on Render's free tier. Also, the free tier goes to sleep after periods of inactivity, so the first request after waking up could be slow.

## Important Notes
- Update your frontend application's API URL to point to your new Render URL (e.g. `https://agriopti-ml-api.onrender.com/predict-crop`) once deployment is successful.
