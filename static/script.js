document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    const loadingState = document.getElementById('loading-state');

    // State elements
    const stateEpisodic = document.getElementById('state-episodic');
    const stateGist = document.getElementById('state-gist');
    const stateEntities = document.getElementById('state-entities');
    const stateGoal = document.getElementById('state-goal');
    const stateConstraints = document.getElementById('state-constraints');
    const stateArtifacts = document.getElementById('state-artifacts');
    const stateFullJson = document.getElementById('state-full-json');

    // Generate a random session ID for this page load
    const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);

    function addMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return contentDiv; // Return for streaming updates
    }

    function updateStateDisplay(state) {
        if (!state) return;

        // Simple fields
        stateEpisodic.textContent = state.episodic_trace || '-';
        stateGist.textContent = state.semantic_gist || '-';
        stateGoal.textContent = state.goal_orientation || '-';

        // Entities (Array)
        stateEntities.innerHTML = '';
        if (state.focal_entities && state.focal_entities.length > 0) {
            state.focal_entities.forEach(entity => {
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = entity;
                stateEntities.appendChild(tag);
            });
        } else {
            stateEntities.textContent = '-';
        }

        // Constraints (Array)
        stateConstraints.innerHTML = '';
        if (state.constraints && state.constraints.length > 0) {
            const ul = document.createElement('ul');
            state.constraints.forEach(c => {
                const li = document.createElement('li');
                li.textContent = c;
                ul.appendChild(li);
            });
            stateConstraints.appendChild(ul);
        } else {
            stateConstraints.textContent = '-';
        }

        // Artifacts (Array)
        stateArtifacts.innerHTML = '';
        if (state.retrieved_artifacts && state.retrieved_artifacts.length > 0) {
            const ul = document.createElement('ul');
            state.retrieved_artifacts.forEach(a => {
                const li = document.createElement('li');
                li.textContent = a;
                ul.appendChild(li);
            });
            stateArtifacts.appendChild(ul);
        } else {
            stateArtifacts.textContent = '-';
        }

        // Full JSON
        stateFullJson.textContent = JSON.stringify(state, null, 2);
    }

    async function fetchState() {
        try {
            const res = await fetch(`/state/${sessionId}`);
            if (res.ok) {
                const state = await res.json();
                updateStateDisplay(state);
            }
        } catch (e) {
            console.error("Failed to fetch state:", e);
        }
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // UI Updates
        addMessage('user', text);
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;
        loadingState.classList.remove('hidden');

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: text,
                    session_id: sessionId
                }),
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }

            // Prepare AI message bubble
            const aiContentDiv = addMessage('ai', '');
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let firstChunkReceived = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                // Hide loading state as soon as we get the first chunk
                if (!firstChunkReceived) {
                    loadingState.classList.add('hidden');
                    firstChunkReceived = true;
                }

                const chunk = decoder.decode(value, { stream: true });
                if (chunk) {
                    aiContentDiv.textContent += chunk;
                    // Use requestAnimationFrame for smooth scrolling
                    requestAnimationFrame(() => {
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    });
                }
            }
            
            // Fetch and update state after streaming is done
            await fetchState();

        } catch (error) {
            console.error(error);
            addMessage('system', 'エラーが発生しました: ' + error.message);
            loadingState.classList.add('hidden'); // Ensure hidden on error
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            if (!loadingState.classList.contains('hidden')) {
                loadingState.classList.add('hidden');
            }
            userInput.focus();
        }
    }

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') this.style.height = '';
    });
});
