document.addEventListener("DOMContentLoaded", () => {
    const placeholder = document.getElementById('navbar-placeholder');
    if (!placeholder) return;

    // Load the navbar HTML from Flask route (bypassing cache)
    fetch('/navbar', { cache: 'no-store' })
        .then(response => response.text())
        .then(html => {
            placeholder.innerHTML = html;
            
            // Set active state based on current URL
            const currentLocation = window.location.pathname;
            const links = placeholder.querySelectorAll('.nav-links a');
            
            links.forEach(link => {
                const linkHref = link.getAttribute('href');
                if (linkHref === currentLocation || 
                    (currentLocation === '/' && linkHref === '/')) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });

            // Initialize scroll effect
            const navbar = placeholder.querySelector('.navbar');
            if (navbar) {
                window.addEventListener('scroll', () => {
                    if (window.scrollY > 30) {
                        navbar.classList.add('scrolled');
                    } else {
                        navbar.classList.remove('scrolled');
                    }
                });
            }

            // --- GLOBAL UX: Theme Toggle ---
            const themeToggleBtn = placeholder.querySelector('#theme-toggle');
            const icon = themeToggleBtn ? themeToggleBtn.querySelector('i') : null;
            
            const currentTheme = localStorage.getItem('theme-preference') || 'light';
            if (currentTheme === 'dark') {
                document.body.classList.add('dark-theme');
                if (icon) {
                    icon.classList.remove('fa-moon');
                    icon.classList.add('fa-sun');
                }
            }

            if (themeToggleBtn && icon) {
                themeToggleBtn.addEventListener('click', () => {
                    document.body.classList.toggle('dark-theme');
                    const isDark = document.body.classList.contains('dark-theme');
                    localStorage.setItem('theme-preference', isDark ? 'dark' : 'light');
                    
                    if (isDark) {
                        icon.classList.remove('fa-moon');
                        icon.classList.add('fa-sun');
                    } else {
                        icon.classList.remove('fa-sun');
                        icon.classList.add('fa-moon');
                    }
                });
            }

            // --- MOBILE MENU TOGGLE ---
            const mobileMenuBtn = placeholder.querySelector('#mobile-menu-btn');
            const navLinks = placeholder.querySelector('.nav-links');
            const mobileIcon = mobileMenuBtn ? mobileMenuBtn.querySelector('i') : null;

            if (mobileMenuBtn && navLinks && mobileIcon) {
                mobileMenuBtn.addEventListener('click', () => {
                    navLinks.classList.toggle('active');
                    
                    if (navLinks.classList.contains('active')) {
                        mobileIcon.classList.remove('fa-ellipsis-vertical');
                        mobileIcon.classList.add('fa-xmark');
                    } else {
                        mobileIcon.classList.remove('fa-xmark');
                        mobileIcon.classList.add('fa-ellipsis-vertical');
                    }
                });

                // Close menu when clicking outside
                document.addEventListener('click', (e) => {
                    if (!e.target.closest('.nav-container') && navLinks.classList.contains('active')) {
                        navLinks.classList.remove('active');
                        mobileIcon.classList.remove('fa-xmark');
                        mobileIcon.classList.add('fa-ellipsis-vertical');
                    }
                });
            }

            // --- GLOBAL UX: Glowing Cursor ---
            const cursorGlow = document.createElement('div');
            cursorGlow.id = 'cursor-glow';
            document.body.appendChild(cursorGlow);
            
            document.addEventListener('mousemove', (e) => {
                // Use requestAnimationFrame for smooth 60fps tracking
                requestAnimationFrame(() => {
                    cursorGlow.style.left = e.clientX + 'px';
                    cursorGlow.style.top = e.clientY + 'px';
                });
            });

            // --- GLOBAL UX: 3D Tilt Cards ---
            const initTilt = () => {
                const tiltElements = document.querySelectorAll('.card, .feature-card, .chart-card, .metric-box, .processing-card');
                
                tiltElements.forEach(el => {
                    // Set smooth initial transition state
                    el.style.transition = 'transform 0.4s cubic-bezier(0.23, 1, 0.32, 1), box-shadow 0.4s ease';

                    el.addEventListener('mousemove', (e) => {
                        const rect = el.getBoundingClientRect();
                        const x = e.clientX - rect.left;
                        const y = e.clientY - rect.top;
                        
                        const centerX = rect.width / 2;
                        const centerY = rect.height / 2;
                        
                        const rotateX = ((y - centerY) / centerY) * -4; // Max 4 deg tilt
                        const rotateY = ((x - centerX) / centerX) * 4;
                        
                        el.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
                    });

                    el.addEventListener('mouseleave', () => {
                        // Reset rotation on leave
                        el.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
                    });
                });
            };
            
            // Allow DOM modifications from other scripts to settle before attaching tilt
            setTimeout(initTilt, 600);
        })
        .catch(error => {
            console.error("Error loading the navbar:", error);
        });

    // ==========================================
    // GLOBAL UX: TOAST NOTIFICATION ENGINE
    // ==========================================
    window.showToast = function(message, type = 'success') {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            document.body.appendChild(container);
        }

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = type === 'success' ? '<i class="fa-solid fa-circle-check"></i>' : '<i class="fa-solid fa-circle-exclamation"></i>';
        toast.innerHTML = `${icon} <span>${message}</span>`;
        
        container.appendChild(toast);
        
        // Trigger reflow for animation
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 400); 
        }, 3000);
    };

    // ==========================================
    // GLOBAL UX: SPA PAGE TRANSITIONS
    // ==========================================
    document.body.classList.remove('page-transitioning');
    
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (!link) return;
        
        const href = link.getAttribute('href');
        if (href && !href.startsWith('#') && link.target !== '_blank' && !href.startsWith('javascript:')) {
            e.preventDefault();
            document.body.classList.add('page-transitioning');
            setTimeout(() => {
                window.location.href = href;
            }, 400);
        }
    });

    // Handle button programmatic redirects
    document.querySelectorAll('button[onclick*="window.location"]').forEach(btn => {
        const onclickRaw = btn.getAttribute('onclick');
        btn.removeAttribute('onclick');
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            document.body.classList.add('page-transitioning');
            setTimeout(() => {
                // simple eval of the onclick
                eval(onclickRaw);
            }, 400);
        });
    });

    // ==========================================
    // GLOBAL UX: MAGNETIC MICRO-INTERACTIONS
    // ==========================================
    function initMagneticButtons() {
        const magneticElements = document.querySelectorAll('.primary-btn, .btn-start, .btn-analyze');
        
        magneticElements.forEach(el => {
            el.classList.add('magnetic-btn');
            
            el.addEventListener('mousemove', (e) => {
                const rect = el.getBoundingClientRect();
                // Center of the button
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                
                // Mouse position relative to center
                const diffX = e.clientX - centerX;
                const diffY = e.clientY - centerY;
                
                // Max pull distance in pixels
                const strength = 12; 
                
                const pullX = (diffX / rect.width) * strength;
                const pullY = (diffY / rect.height) * strength;
                
                el.style.transform = `translate(${pullX}px, ${pullY}px) scale(1.02)`;
            });
            
            el.addEventListener('mouseleave', () => {
                el.style.transform = `translate(0px, 0px) scale(1)`;
            });
        });
    }

    setTimeout(initMagneticButtons, 500);

});
