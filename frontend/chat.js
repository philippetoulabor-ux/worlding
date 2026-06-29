(function () {
  const CHAT_API =
    document.querySelector('meta[name="zettelkasten-api"]')?.content || '/api/chat';

  const fab = document.getElementById('chat-fab');
  const panel = document.getElementById('chat-panel');
  const messagesEl = document.getElementById('chat-messages');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');

  let isOpen = false;
  let isStreaming = false;

  fab.addEventListener('click', () => {
    isOpen = !isOpen;
    panel.classList.toggle('hidden', !isOpen);
    if (isOpen) input.focus();
  });

  function appendMessage(role, text) {
    const el = document.createElement('div');
    el.className = `chat-msg ${role}`;
    el.textContent = text;
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  function appendSources(ids, titles) {
    if (!ids.length) return;

    const wrap = document.createElement('div');
    wrap.className = 'chat-sources';
    wrap.textContent = 'Quellen: ';

    ids.forEach((id, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = titles[i] || id;
      btn.addEventListener('click', () => {
        if (typeof window.openNoteById === 'function') {
          window.openNoteById(id);
        }
      });
      wrap.appendChild(btn);
    });

    messagesEl.lastElementChild?.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    if (typeof window.highlightChatSources === 'function') {
      window.highlightChatSources(ids);
    }
  }

  function showTyping() {
    const el = document.createElement('div');
    el.className = 'chat-typing';
    el.id = 'chat-typing';
    el.textContent = 'Denke nach…';
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function hideTyping() {
    document.getElementById('chat-typing')?.remove();
  }

  async function sendMessage(text) {
    if (isStreaming) return;

    isStreaming = true;
    sendBtn.disabled = true;
    appendMessage('user', text);
    input.value = '';
    showTyping();

    let assistantEl = null;

    try {
      const response = await fetch(CHAT_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Fehler ${response.status}`);
      }

      hideTyping();
      assistantEl = appendMessage('assistant', '');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const event = JSON.parse(line.slice(6));

          if (event.type === 'token') {
            assistantEl.textContent += event.content;
            messagesEl.scrollTop = messagesEl.scrollHeight;
          } else if (event.type === 'sources') {
            appendSources(event.ids || [], event.titles || []);
          } else if (event.type === 'error') {
            throw new Error(event.content);
          }
        }
      }
    } catch (error) {
      hideTyping();
      if (assistantEl) assistantEl.remove();
      appendMessage('error', error.message || 'Anfrage fehlgeschlagen.');
    } finally {
      isStreaming = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (text) sendMessage(text);
  });

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
})();
