import os
import json

# Define the compare method
compare_method = 'codef'
videos_a = set(os.listdir(compare_method))
videos_b = set(os.listdir('StreamV2V'))

# Find common videos
common_videos = sorted(videos_a.intersection(videos_b))


# Function to load JSON data from a file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

file_path = 'eval.json'  # Path to your JSON file

json_data = load_data(file_path)

prompt_dict = {}
for data in json_data:
    prompt_dict[data['vid_name']] = data['prompt']


# HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Video Comparison</title>
    <style>
        .video-container { display: none; }
        .active { display: block; }
    </style>
    <script>
        let currentVideoIndex = -1;
        let responses = {};

        function collectResponses() {
            const videoName = document.querySelectorAll('.video-container')[currentVideoIndex].getAttribute('data-video');
            responses[videoName] = {
                consistency: document.querySelector(`input[name="${videoName}_consistency"]:checked`) ? document.querySelector(`input[name="${videoName}_consistency"]:checked`).value : 'No response',
                alignment: document.querySelector(`input[name="${videoName}_alignment"]:checked`) ? document.querySelector(`input[name="${videoName}_alignment"]:checked`).value : 'No response',
                preference: document.querySelector(`input[name="${videoName}_preference"]:checked`) ? document.querySelector(`input[name="${videoName}_preference"]:checked`).value : 'No response',
            };
        }

        function displayResults() {
            let results = 'Comparison Results:\\n';
            for (const [video, response] of Object.entries(responses)) {
                results += `${video}: Consistency - ${response.consistency}, Alignment - ${response.alignment}, Preference - ${response.preference}\n`;
            }
            alert(results);

            // Create a blob from the responses object
            const blob = new Blob([JSON.stringify(responses, null, 2)], { type: 'application/json' });

            // Create a link element
            const a = document.createElement('a');

            // Set the download attribute to the name of the file you want to download
            a.download = 'results.json';

            // Create a URL for the blob
            a.href = window.URL.createObjectURL(blob);

            // Append the link to the body
            document.body.appendChild(a);

            // Programmatically click the link to trigger the download
            a.click();

            // Remove the link from the body
            document.body.removeChild(a);
        }

        function nextVideo() {
            if (currentVideoIndex >= 0) {
                collectResponses();
            }
            currentVideoIndex++;
            const containers = document.querySelectorAll('.video-container');
            if (currentVideoIndex < containers.length) {
                containers[currentVideoIndex].style.display = 'block';
                if (currentVideoIndex > 0) containers[currentVideoIndex - 1].style.display = 'none';
                containers[currentVideoIndex].querySelectorAll('video').forEach(video => video.play());
            } else {
                displayResults();
            }
        }

        document.addEventListener('DOMContentLoaded', (event) => {
            nextVideo(); // Show the first video on load
        });
    </script>
</head>
<body>
    <h1>Video Comparison</h1>
"""

# Add video containers with autoplay and radio button questions
for video in common_videos:
    video_name = video.split('.')[0]
    prompt = prompt_dict.get(video_name, "No prompt available")  # Get the prompt for the video, defaulting to a placeholder if not found
    html_content += f"""
    <div class="video-container" data-video="{video}">
        <h2>{video}</h2>
        <p><strong>Prompt:</strong> {prompt}</p>  <!-- Display the prompt here -->
        <video controls autoplay muted loop src="{compare_method}/{video}"></video>
        <video controls autoplay muted loop src="StreamV2V/{video}"></video>
        <div>
            <p>1. Which one is more consistent:</p>
            <label><input type="radio" name="{video}_consistency" value="A"> A</label>
            <label><input type="radio" name="{video}_consistency" value="B"> B</label>
            <label><input type="radio" name="{video}_consistency" value="Draw"> Draw</label>
            <p>2. Which one aligns better with the prompt:</p>
            <label><input type="radio" name="{video}_alignment" value="A"> A</label>
            <label><input type="radio" name="{video}_alignment" value="B"> B</label>
            <label><input type="radio" name="{video}_alignment" value="Draw"> Draw</label>
            <p>3. Which one do you think is better:</p>
            <label><input type="radio" name="{video}_preference" value="A"> A</label>
            <label><input type="radio" name="{video}_preference" value="B"> B</label>
            <label><input type="radio" name="{video}_preference" value="Draw"> Draw</label>
        </div>
    </div>
    """

# Close HTML with the Next button
html_content += """
    <button onclick="nextVideo()">Next Video</button>
</body>
</html>
"""

# Write HTML file
html_filename = f"video_comparison_{compare_method}.html"
with open(html_filename, "w") as file:
    file.write(html_content)

print(f"HTML file generated successfully. Please serve this file from a local server and open {html_filename} in your browser.")

