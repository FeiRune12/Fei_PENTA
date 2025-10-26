document.getElementById('image-form').addEventListener('submit', async (event) => {
  event.preventDefault();

  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return alert("Por favor, digite um pedido!");

  const loader = document.getElementById('loader');
  const resultSection = document.getElementById('result-section');
  const resultImage = document.getElementById('result-image');

  loader.classList.remove('hidden');
  resultSection.classList.add('hidden');

  try {
    const response = await fetch('https://fei-penta.onrender.com/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) throw new Error('Erro na geração da imagem');

    const data = await response.json();
    resultImage.src = data.image;
    resultSection.classList.remove('hidden');
  } catch (error) {
    alert('Falha ao gerar imagem: ' + error.message);
  } finally {
    loader.classList.add('hidden');
  }
});
