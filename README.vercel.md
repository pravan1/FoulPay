# Vercel Deployment Guide for FoulPay

## Deployment Steps

### 1. Backend Deployment (API)

1. Navigate to your Vercel dashboard
2. Create a new project and import your GitHub repository
3. In the project settings, set the **Root Directory** to `backend`
4. Add the following environment variables:
   - `MONGODB_URI`: Your MongoDB connection string
   - `FIREBASE_SERVICE_ACCOUNT_KEY`: Your Firebase service account key (as JSON string)
   - `PYTHONPATH`: Set to `/var/task/backend`

5. Deploy the backend

### 2. Frontend Deployment

1. Create another new project in Vercel
2. Import the same GitHub repository
3. Leave the **Root Directory** as empty (it will use the root `vercel.json`)
4. Add the following environment variables:
   - `REACT_APP_API_URL`: Your backend Vercel URL (e.g., `https://your-backend-project.vercel.app`)
   - `REACT_APP_AUTH0_DOMAIN`: Your Auth0 domain
   - `REACT_APP_AUTH0_CLIENT_ID`: Your Auth0 client ID

5. Deploy the frontend

## Configuration Files

The project includes the following Vercel configuration files:

- `/vercel.json` - Frontend configuration
- `/backend/vercel.json` - Backend API configuration
- `/backend/api/index.py` - Vercel Python runtime entry point
- `/package.json` - Root package.json for build commands
- `/.env.example` - Example environment variables

## Important Notes

1. **CORS Configuration**: Update the CORS origins in `backend/main.py` to include your Vercel frontend URL
2. **API Routes**: The frontend is configured to proxy `/api/*` requests to your backend URL
3. **Environment Variables**: Ensure all required environment variables are set in both deployments
4. **Python Dependencies**: All Python packages in `requirements.txt` will be automatically installed

## Testing Locally

Before deploying, test your configuration:

```bash
# Frontend
cd frontend
npm install
npm start

# Backend (in another terminal)
cd backend
pip install -r requirements.txt
python main.py
```

## Troubleshooting

1. **Module Import Errors**: The `backend/api/index.py` file handles Python path configuration
2. **CORS Issues**: Make sure to update allowed origins in `backend/main.py`
3. **Environment Variables**: Double-check all environment variables are properly set in Vercel dashboard