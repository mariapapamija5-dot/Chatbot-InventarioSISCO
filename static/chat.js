// ══════════════════════════════════════════════════════════════
// SISCO — Chat conversacional con modelo híbrido BETO + Likert
// ══════════════════════════════════════════════════════════════

// ── Elementos del DOM ──────────────────────────────────
const chatMessages  = document.getElementById('chatMessages');
const userInput     = document.getElementById('userInput');
const sendBtn       = document.getElementById('sendBtn');
const wordCounter   = document.getElementById('wordCounter');
const nivelCard     = document.getElementById('nivelCard');
const nivelBadge    = document.getElementById('nivelBadge');
const nivelDesc     = document.getElementById('nivelDesc');
const progressCard  = document.getElementById('progressCard');
const progressBar   = document.getElementById('progressBar');
const progressText  = document.getElementById('progressText');
const nuevoContainer = document.getElementById('nuevoContainer');

const MIN_PALABRAS = 5;

// ── Contador de palabras ───────────────────────────────
userInput.addEventListener('input', () => {
  const palabras = userInput.value.trim().split(/\s+/).filter(w => w.length > 0);
  const n = palabras.length;
  wordCounter.textContent = `${n} palabra${n !== 1 ? 's' : ''}`;
  wordCounter.classList.toggle('ok', n >= MIN_PALABRAS);
  sendBtn.disabled = n < MIN_PALABRAS;
});

// Enter para enviar (Shift+Enter = nueva línea)
userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) enviarMensaje();
  }
});

// ── Agregar mensaje al chat ────────────────────────────
function agregarMensaje(texto, tipo, claseExtra = '') {
  const div = document.createElement('div');
  div.className = `msg msg-${tipo} ${claseExtra}`.trim();

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = tipo === 'user' ? '👤' : '🎓';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  if (texto.startsWith('<')) {
    bubble.innerHTML = texto;
  } else {
    bubble.innerHTML = texto.split('\n').map(l => `<p>${l}</p>`).join('');
  }

  div.appendChild(avatar);
  div.appendChild(bubble);
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

// ── Indicador de escritura (typing) ───────────────────
function mostrarTyping() {
  const div = document.createElement('div');
  div.className = 'msg msg-bot typing';
  div.id = 'typingIndicator';
  div.innerHTML = `
    <div class="msg-avatar">🎓</div>
    <div class="msg-bubble">
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function quitarTyping() {
  const t = document.getElementById('typingIndicator');
  if (t) t.remove();
}

// ── Actualizar barra de progreso ───────────────────────
function actualizarProgreso(paso, total) {
  progressCard.style.display = 'block';
  const pct = Math.round((paso / total) * 100);
  progressBar.style.width = `${pct}%`;
  progressText.textContent = `Paso ${paso} de ${total}`;
}

// ── Mostrar resultado en sidebar ───────────────────────
function mostrarNivelSidebar(nivel) {
  const descripciones = {
    bajo:     'Tu nivel de estrés es manejable. Sigue aplicando tus estrategias de afrontamiento.',
    moderado: 'Presentas un nivel moderado de estrés. Organiza tus tiempos y busca apoyo si lo necesitas.',
    alto:     'Tu nivel de estrés es alto. Te recomendamos acudir a bienestar universitario.'
  };
  nivelBadge.textContent = nivel.charAt(0).toUpperCase() + nivel.slice(1);
  nivelBadge.className   = `nivel-badge ${nivel}`;
  nivelDesc.textContent  = descripciones[nivel] || '';
  nivelCard.style.display = 'block';
  nivelCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Construir respuesta del bot con resultado ──────────
function construirRespuestaBot(nivel, confianza, probas, recomendacion) {
  const emoji  = { bajo: '🟢', moderado: '🟡', alto: '🔴' };
  const titulo = { bajo: 'Nivel bajo', moderado: 'Nivel moderado', alto: 'Nivel alto' };

  // Barras de probabilidad
  let barrasHTML = '';
  const colores = { bajo: '#2d8a6e', moderado: '#b45309', alto: '#b91c1c' };
  for (const [clase, prob] of Object.entries(probas)) {
    const pct = (prob * 100).toFixed(1);
    barrasHTML += `
      <div class="prob-row">
        <span class="prob-label">${clase.charAt(0).toUpperCase() + clase.slice(1)}</span>
        <div class="prob-bar-container">
          <div class="prob-bar" style="width:${pct}%; background:${colores[clase]}"></div>
        </div>
        <span class="prob-value">${pct}%</span>
      </div>`;
  }

  return `
    <p>${emoji[nivel]} <strong>${titulo[nivel]} de estrés académico</strong></p>
    <p>${recomendacion}</p>
    <div class="prob-section">
      <p class="prob-title">Probabilidades del modelo:</p>
      ${barrasHTML}
    </div>
    <p style="font-size:.8rem;color:#64748b;margin-top:.5rem;">
      Confianza del modelo: ${(confianza * 100).toFixed(1)}%
    </p>`;
}

// ── Enviar mensaje ─────────────────────────────────────
async function enviarMensaje() {
  const texto = userInput.value.trim();
  if (!texto) return;

  // Mostrar mensaje del usuario
  agregarMensaje(texto, 'user');
  userInput.value = '';
  wordCounter.textContent = '0 palabras';
  wordCounter.classList.remove('ok');
  sendBtn.disabled = true;
  userInput.disabled = true;

  // Mostrar typing
  mostrarTyping();

  try {
    const response = await fetch('/predecir', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texto })
    });

    const data = await response.json();
    quitarTyping();

    if (data.error) {
      agregarMensaje(data.error, 'bot');
      userInput.disabled = false;
      userInput.focus();
      return;
    }

    if (data.tipo === 'pregunta') {
      // Mostrar pregunta de seguimiento
      agregarMensaje(data.pregunta, 'bot');
      actualizarProgreso(data.paso, data.total_pasos);
      userInput.disabled = false;
      userInput.focus();

    } else if (data.tipo === 'resultado') {
      // Mostrar resultado final
      const claseExtra = `msg-resultado nivel-${data.nivel}`;
      agregarMensaje(
        construirRespuestaBot(data.nivel, data.confianza, data.probas, data.recomendacion),
        'bot',
        claseExtra
      );

      // Actualizar sidebar
      mostrarNivelSidebar(data.nivel);

      // Completar progreso
      progressBar.style.width = '100%';
      progressText.textContent = 'Evaluación completada';

      // Mostrar botón de nuevo análisis
      nuevoContainer.style.display = 'flex';

      // Deshabilitar input
      userInput.disabled = true;
      userInput.placeholder = 'Evaluación completada. Presiona "Nuevo análisis" para comenzar de nuevo.';
    }

  } catch (error) {
    quitarTyping();
    agregarMensaje('No se pudo conectar con el servidor. Verifica tu conexión e intenta de nuevo.', 'bot');
    userInput.disabled = false;
  }
}

// ── Reiniciar chat ─────────────────────────────────────
async function reiniciarChat() {
  try {
    await fetch('/reiniciar', { method: 'POST' });
  } catch (e) {
    // Continuar de todas formas
  }

  // Limpiar mensajes (dejar solo el saludo inicial)
  const mensajes = chatMessages.querySelectorAll('.msg');
  mensajes.forEach((msg, i) => {
    if (i > 0) msg.remove();
  });

  // Ocultar resultado y progreso
  nivelCard.style.display = 'none';
  progressCard.style.display = 'none';
  progressBar.style.width = '0%';
  nuevoContainer.style.display = 'none';

  // Rehabilitar input
  userInput.disabled = false;
  userInput.value = '';
  userInput.placeholder = 'Escribe aquí cómo te has sentido este semestre...';
  wordCounter.textContent = '0 palabras';
  wordCounter.classList.remove('ok');
  sendBtn.disabled = true;
  userInput.focus();
}
