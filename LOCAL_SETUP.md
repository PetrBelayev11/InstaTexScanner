# 🚀 InstaTexScanner Local Launch

## Quick Start

### Windows
Just run the file:
```bash
start_local.bat
```

### Linux/Mac
```bash
chmod +x start_local.sh
./start_local.sh
```

## Manual Launch

### 1. Install Dependencies
```bash
py -m pip install -r code/deployment/api/requirements.txt
py -m pip install -r code/models/requirements.txt
```

### 2. Launch API Server
In the first terminal:
```bash
py run_api.py
```

API will be available at: http://localhost:8000
API Documentation: http://localhost:8000/docs

### 3. Launch Frontend Server
In the second terminal:
```bash
py run_frontend.py
```

Frontend will be available at: http://localhost:3000

## Project Structure

- `run_api.py` - Script to launch the API server
- `run_frontend.py` - Script to launch the frontend server
- `start_local.bat` - Automatic launch for Windows
- `start_local.sh` - Automatic launch for Linux/Mac
- `code/deployment/api/main_local.py` - Local version of API (adapted for running without Docker)

## Ports

- **API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## Notes

- Make sure ports 8000 and 3000 are free
- To stop servers, press `Ctrl+C` in each terminal
- All uploaded files are saved in the `shared_data/` folder

## Troubleshooting

### Port Already in Use
If a port is busy, change the port number in the corresponding files:
- API: `run_api.py` (line with `port=8000`)
- Frontend: `run_frontend.py` (line with `PORT = 3000`)

### Import Errors
Make sure you are in the root directory of the project when running scripts.

### Models Not Found
Some functions may require pretrained models. Ensure that model files are in the `models/` folder.

