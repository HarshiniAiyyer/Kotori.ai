# Deployment Checklist

## Backend (Render)
- [x] Create `requirements.txt` with all Python dependencies.
- [ ] Ensure `render.yaml` is correctly configured for the backend service.
- [ ] Set up environment variables on Render (HUGGINGFACE_API_TOKEN, GROQ_API_KEY, CHROMA_DB_PATH, DATA_DIR).
- [ ] Verify `uvicorn` command in `render.yaml` is correct.
- [ ] Test API endpoints after deployment.

## Frontend (Vercel)
- [ ] Ensure `package.json` has correct build and start scripts.
- [ ] Configure Vercel project settings (build command, output directory).
- [ ] Set up environment variables on Vercel (e.g., API_BASE_URL).
- [ ] Test frontend application after deployment.

## General
- [ ] Review `.gitignore` to ensure sensitive files are not committed.
- [ ] Update `README.md` with deployment instructions.
- [ ] Verify all necessary data files are accessible (e.g., PDFs in `data` folder).