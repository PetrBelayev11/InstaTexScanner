let currentFile = null;

// Get API URL
const API_BASE_URL = 'http://localhost:8000';

// File input handling
document.getElementById('fileInput').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and drop handling
const dropZone = document.getElementById('dropZone');

dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', function() {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showResult('Please select an image file (JPEG, PNG, etc.)', 'error');
        return;
    }
    
    currentFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').style.display = 'block';
    hideResult();
}

async function convert(format) {
    if (!currentFile) {
        showResult('Please select a file first', 'error');
        return;
    }
    
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    
    loading.style.display = 'block';
    result.style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // Use query parameter as your backend expects
        const response = await fetch(`${API_BASE_URL}/convert?output_format=${format}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Conversion failed');
        }
        
        const data = await response.json();
        
        if (data.success) {
            if (format === 'pdf') {
                // Download PDF file
                const downloadUrl = `${API_BASE_URL}${data.download_url}`;
                showResult(`
                    <div class="success">
                        <h3>✅ ${data.message}</h3>
                        <button onclick="window.open('${downloadUrl}', '_blank')">📥 Download PDF</button>
                    </div>
                `, 'success');
            } else if (format === 'text') {
                // Handle text extraction response
                const downloadUrl = `${API_BASE_URL}${data.download_url}`;
                showResult(`
                    <div class="success">
                        <h3>✅ ${data.message}</h3>
                        <div style="margin: 15px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; max-height: 200px; overflow-y: auto;">
                            <h4>Extracted Text:</h4>
                            <pre style="white-space: pre-wrap; font-family: inherit;">${data.text || 'No text extracted'}</pre>
                        </div>
                        <button onclick="window.open('${downloadUrl}', '_blank')">📥 Download Text File</button>
                    </div>
                `, 'success');
            } else if (format === 'latex') {
                // Handle LaTeX response
                const downloadUrl = `${API_BASE_URL}${data.download_url}`;
                showResult(`
                    <div class="success">
                        <h3>✅ ${data.message}</h3>
                        <div style="margin: 15px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; max-height: 200px; overflow-y: auto;">
                            <h4>Extracted LaTeX:</h4>
                            <pre style="white-space: pre-wrap; font-family: inherit;">${data.latex || 'No LaTeX extracted'}</pre>
                        </div>
                        <button onclick="window.open('${downloadUrl}', '_blank')">📥 Download LaTeX File</button>
                    </div>
                `, 'success');
            }
        } else {
            showResult('Conversion failed: ' + data.message, 'error');
        }
    } catch (error) {
        showResult('Error: ' + error.message, 'error');
        console.error('Error:', error);
    } finally {
        loading.style.display = 'none';
    }
}

function showResult(message, type) {
    const result = document.getElementById('result');
    result.innerHTML = message;
    result.className = `result ${type}`;
    result.style.display = 'block';
}

function hideResult() {
    document.getElementById('result').style.display = 'none';
}

// Check backend health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('Backend is connected');
        }
    } catch (error) {
        console.warn('Backend is not reachable:', error.message);
        showResult('Warning: Backend server is not reachable', 'error');
    }
}

// Check health on page load
document.addEventListener('DOMContentLoaded', checkHealth);