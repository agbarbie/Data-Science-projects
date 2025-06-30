async function predict() {
    const inputs = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];
    const values = inputs.map(id => document.getElementById(id).value);

    if (values.some(val => !val)) {
        showResult('<div class="error">Please fill in all fields</div>');
        return;
    }

    const btn = document.querySelector('.predict-btn');
    const btnText = document.getElementById('btn-text');
    btnText.innerHTML = '<span class="loading"></span> Analyzing...';
    btn.disabled = true;

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: values.map(parseFloat) })
        });

        const result = await response.json();

        if (response.ok) {
            const emoji = getSpeciesEmoji(result.prediction);
            const confidence = (result.confidence * 100).toFixed(1);

            showResult(`
                <div class="species">${emoji} ${result.prediction.toUpperCase()}</div>
                <div class="confidence">${confidence}% confidence</div>
            `);
        } else {
            showResult(`<div class="error">Error: ${result.error}</div>`);
        }
    } catch (error) {
        showResult(`<div class="error">Connection failed: ${error.message}</div>`);
    } finally {
        btnText.innerHTML = 'Predict Species';
        btn.disabled = false;
    }
}

function showResult(html) {
    const result = document.getElementById('result');
    result.innerHTML = html;
    result.classList.add('show');
}

function getSpeciesEmoji(species) {
    const emojis = { setosa: '🌸', versicolor: '🌺', virginica: '🌻' };
    return emojis[species] || '🌼';
}

async function loadMetrics() {
    try {
        const response = await fetch('http://localhost:5000/performance');
        const data = await response.json();

        document.getElementById('accuracy').textContent = (data.test_accuracy * 100).toFixed(1) + '%';
        document.getElementById('loss').textContent = data.test_loss.toFixed(3);
        document.getElementById('params').textContent = data.total_parameters;

        const species = ['setosa', 'versicolor', 'virginica'];
        const container = document.getElementById('class-accuracies');

        container.innerHTML = species.map(name => {
            const accuracy = data.per_class_accuracy[name] * 100;
            const emoji = getSpeciesEmoji(name);

            return `
                <div class="class-accuracy">
                    <div class="class-name">${emoji} ${name}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${accuracy}%"></div>
                    </div>
                    <div class="accuracy-value">${accuracy.toFixed(1)}%</div>
                </div>
            `;
        }).join('');

    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

window.onload = loadMetrics;
