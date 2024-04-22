if (!window.dash_clientside) { window.dash_clientside = {}; }
window.dash_clientside.clientside = {
    scrollToBottom: function() {
        setTimeout(() => {
            const element = document.getElementById('chat-area');
            if (element) {
                element.scrollTop = element.scrollHeight;
            }
        }, 0);
    }
};
