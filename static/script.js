/* ========== INIT LUCIDE ICONS ========== */
lucide.createIcons();

/* ========== NAVBAR SCROLL EFFECT ========== */
const navbar = document.querySelector(".navbar");
let lastScroll = 0;

window.addEventListener("scroll", () => {
  if (window.scrollY > 50) {
    navbar.classList.add("scrolled");
  } else {
    navbar.classList.remove("scrolled");
  }
  lastScroll = window.scrollY;
});

/* ========== COUNTER ANIMATION (EASED) ========== */
const counters = document.querySelectorAll(".counter");

function animateCounter(counter) {
  const target = +counter.dataset.target;
  const duration = 1800;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // Ease-out cubic for smooth deceleration
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.floor(eased * target);

    counter.innerText = current + "%";

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      counter.innerText = target + "%";
    }
  }

  requestAnimationFrame(update);
}

// Use IntersectionObserver to trigger counters when visible
const counterObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.5 }
);

counters.forEach((counter) => counterObserver.observe(counter));

/* ========== SCROLL REVEAL (STAGGERED) ========== */
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
);

document.querySelectorAll(".reveal").forEach((el) => {
  revealObserver.observe(el);
});

/* ========== TYPED TEXT EFFECT ========== */
function typeText(element, text, speed = 70) {
  element.textContent = "";
  element.style.borderRight = "2px solid #22c55e";
  let i = 0;

  function type() {
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
      setTimeout(type, speed);
    } else {
      // Blink cursor then remove
      setTimeout(() => {
        element.style.borderRight = "none";
      }, 1500);
    }
  }

  type();
}

// Trigger typing on the hero span
const heroSpan = document.querySelector(".hero h1 span");
if (heroSpan) {
  const originalText = heroSpan.textContent;
  heroSpan.textContent = "";

  const heroObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setTimeout(() => typeText(heroSpan, originalText, 65), 600);
          heroObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.3 }
  );

  heroObserver.observe(heroSpan);
}

/* ========== SECTION HEADINGS REVEAL ========== */
const headingObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1";
        entry.target.style.transform = "translateY(0)";
        headingObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.2 }
);

document
  .querySelectorAll(".features h2, .features .subtitle, .steps h2, .steps .subtitle, .cta h2, .cta p")
  .forEach((el) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(25px)";
    el.style.transition = "opacity 0.7s ease, transform 0.7s ease";
    headingObserver.observe(el);
  });

/* ========== MOBILE MENU TOGGLE ========== */
const toggle = document.querySelector(".menu-toggle");
const navLinks = document.querySelector(".nav-links");

toggle.addEventListener("click", () => {
  navLinks.style.display =
    navLinks.style.display === "flex" ? "none" : "flex";
});

/* ========== TILT EFFECT ON RISK CARD ========== */
const riskCard = document.querySelector(".risk-card");

if (riskCard) {
  riskCard.addEventListener("mousemove", (e) => {
    const rect = riskCard.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = ((y - centerY) / centerY) * -5;
    const rotateY = ((x - centerX) / centerX) * 5;

    riskCard.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
  });

  riskCard.addEventListener("mouseleave", () => {
    riskCard.style.transform = "";
  });
}