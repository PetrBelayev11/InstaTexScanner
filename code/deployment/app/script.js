let currentFile = null;

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
        
        const response = await fetch(`http://localhost:8000/convert?output_format=${format}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            let resultHTML = `
                <div class="success">
                    <h3>✅ ${data.message}</h3>
                    <button onclick="downloadFile('${data.download_url}')">📥 Download ${format.toUpperCase()}</button>
            `;
            
            if (data.text_preview) {
                resultHTML += `
                    <div class="preview">
                        <strong>Text Preview:</strong><br>
                        ${data.text_preview}
                    </div>
                `;
            }
            
            resultHTML += `</div>`;
            showResult(resultHTML, 'success');
        } else {
            showResult('Conversion failed: ' + data.message, 'error');
        }
    } catch (error) {
        showResult('Error: ' + error.message, 'error');
    } finally {
        loading.style.display = 'none';
    }
}

function downloadFile(downloadUrl) {
    window.open(`http://localhost:8000${downloadUrl}`, '_blank');
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