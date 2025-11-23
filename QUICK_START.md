# 🚀 Quick Start InstaTexScanner

## ✅ Project Configured for Local Launch!

### Method 1: Automatic Launch (Recommended)

**Windows:**
```bash
start_local.bat
```

**Linux/Mac:**
```bash
chmod +x start_local.sh
./start_local.sh
```

### Method 2: Manual Launch

#### Step 1: Install Missing Dependencies
```bash
py -m pip install aiofiles uvicorn python-multipart
```

#### Step 2: Launch API Server
In the first terminal:
```bash
py run_api.py
```

#### Step 3: Launch Frontend Server
In the second terminal:
```bash
py run_frontend.py
```

## 🌐 App Access

After starting the servers:

- **Frontend (Web Interface)**: http://localhost:3000
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

## 📝 Notes

1. **Ports**: Make sure ports 8000 and 3000 are free
2. **ML Models**: Large models (TrOCR) may be downloaded when using OCR functions for the first time, this may take time
3. **Stopping**: Press `Ctrl+C` in each terminal to stop servers

## 🔧 Troubleshooting

### "ModuleNotFoundError" Error
Install missing modules:
```bash
py -m pip install -r code/deployment/api/requirements.txt
```

### Port Already in Use
Change ports in files:
- `run_api.py` - port 8000
- `run_frontend.py` - port 3000

### Models Not Loading
Some functions require pretrained models. They will be automatically downloaded from Hugging Face on first use.

## 📁 File Structure

- `run_api.py` - Launch API server
- `run_frontend.py` - Launch frontend server  
- `start_local.bat` / `start_local.sh` - Automatic launch
- `code/deployment/api/main_local.py` - Local version of API
- `LOCAL_SETUP.md` - Detailed documentation

## ✅ Status

- ✅ Launch scripts created
- ✅ Frontend server running on port 3000
- ✅ API server configured to run on port 8000
- ✅ Dependencies installed

**Project is ready to use!**

