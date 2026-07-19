window.addEventListener("DOMContentLoaded", function () {
    const panel = document.getElementById("landingPanel");
    const features = document.getElementById("landingFeatures");

    // Fade-in and zoom the glassmorphism panel as the car finishes driving
    setTimeout(function () {
        if (panel) {
            panel.classList.add("show");
        }
    }, 2200);

    // Fade-in the bottom feature bar shortly after
    setTimeout(function () {
        if (features) {
            features.classList.add("show");
        }
    }, 2500);
});