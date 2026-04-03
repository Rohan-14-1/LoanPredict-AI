/**
 * SmartCredit AI — Ultra-Premium Particle Background System
 * Lightweight, performant canvas particle animation with mouse interactivity
 */
(function () {
    'use strict';

    const CONFIG = {
        particleCount: 45,
        maxConnectionDist: 140,
        mouseRadius: 180,
        baseSpeed: 0.3,
        particleMinSize: 1.2,
        particleMaxSize: 2.8,
        fps: 60
    };

    let canvas, ctx, particles = [], mouse = { x: -500, y: -500 }, w, h, animId;

    function getColors() {
        const isDark = document.body.classList.contains('dark-theme');
        return {
            particle: isDark ? 'rgba(52, 211, 153, 0.5)' : 'rgba(34, 197, 94, 0.35)',
            particleAlt: isDark ? 'rgba(96, 165, 250, 0.4)' : 'rgba(59, 130, 246, 0.25)',
            line: isDark ? 'rgba(52, 211, 153,' : 'rgba(34, 197, 94,',
            lineAlt: isDark ? 'rgba(96, 165, 250,' : 'rgba(59, 130, 246,'
        };
    }

    class Particle {
        constructor() {
            this.reset();
            this.useAlt = Math.random() > 0.6;
        }

        reset() {
            this.x = Math.random() * w;
            this.y = Math.random() * h;
            this.size = CONFIG.particleMinSize + Math.random() * (CONFIG.particleMaxSize - CONFIG.particleMinSize);
            this.speedX = (Math.random() - 0.5) * CONFIG.baseSpeed;
            this.speedY = (Math.random() - 0.5) * CONFIG.baseSpeed;
            this.opacity = 0.3 + Math.random() * 0.5;
            this.pulseSpeed = 0.005 + Math.random() * 0.01;
            this.pulsePhase = Math.random() * Math.PI * 2;
        }

        update() {
            // Mouse attraction
            const dx = mouse.x - this.x;
            const dy = mouse.y - this.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < CONFIG.mouseRadius) {
                const force = (CONFIG.mouseRadius - dist) / CONFIG.mouseRadius;
                this.x += dx * force * 0.008;
                this.y += dy * force * 0.008;
            }

            this.x += this.speedX;
            this.y += this.speedY;
            this.pulsePhase += this.pulseSpeed;

            // Wrap around edges
            if (this.x < -10) this.x = w + 10;
            if (this.x > w + 10) this.x = -10;
            if (this.y < -10) this.y = h + 10;
            if (this.y > h + 10) this.y = -10;
        }

        draw(colors) {
            const pulse = Math.sin(this.pulsePhase) * 0.3 + 0.7;
            const r = this.size * pulse;
            ctx.beginPath();
            ctx.arc(this.x, this.y, r, 0, Math.PI * 2);
            ctx.fillStyle = this.useAlt ? colors.particleAlt : colors.particle;
            ctx.fill();
        }
    }

    function drawConnections(colors) {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONFIG.maxConnectionDist) {
                    const opacity = (1 - dist / CONFIG.maxConnectionDist) * 0.15;
                    const lineColor = particles[i].useAlt ? colors.lineAlt : colors.line;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = lineColor + opacity + ')';
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        const colors = getColors();
        ctx.clearRect(0, 0, w, h);
        particles.forEach(p => { p.update(); p.draw(colors); });
        drawConnections(colors);
        animId = requestAnimationFrame(animate);
    }

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }

    function init() {
        // Avoid duplicate init
        if (document.getElementById('particles-canvas')) return;

        canvas = document.createElement('canvas');
        canvas.id = 'particles-canvas';
        canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:0;transition:opacity 1.5s ease;';
        document.body.prepend(canvas);

        ctx = canvas.getContext('2d');
        resize();

        for (let i = 0; i < CONFIG.particleCount; i++) {
            particles.push(new Particle());
        }

        window.addEventListener('resize', resize);
        document.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        // Fade in the canvas
        requestAnimationFrame(() => { canvas.style.opacity = '1'; });
        animate();
    }

    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
