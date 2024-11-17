// Add event listeners to enhance the user experience
document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.querySelector("#file");
    const uploadButton = document.querySelector("button[type='submit']");

    // Add change event to display selected file name
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            alert("File selected: " + fileInput.files[0].name);
        }
    });

    // Disable the button if no file is selected
    fileInput.addEventListener("input", function () {
        uploadButton.disabled = fileInput.files.length === 0;
    });

    // Add loading animation on form submission
    const form = document.querySelector("form");
    form.addEventListener("submit", function () {
        uploadButton.textContent = "Uploading...";
        uploadButton.disabled = true;
    });
});
