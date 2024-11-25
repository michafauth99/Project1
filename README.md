
<a id="readme-top"></a>




<!-- PROJECT LOGO -->
<br />
<div align="left">
    <img src="images/banner.png" alt="Banner" >
</div>

<h3 align="left">Identifying and counting humans at beaches</h3>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Setup">Set up project</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this project, we aim to develop a computer vision system to estimate the number of people present on a beach, a task commonly known as "crowd counting." Lifeguards typically monitor beaches during summer, gathering data on occupancy, sea conditions, wind, and more. By applying image processing techniques, we can automate the counting of beachgoers, reducing manual efforts and improving monitoring accuracy. This project will encompass all key stages of an image processing workflow, including data annotation, algorithm design and implementation, and result validation.




### Built With
* [![Python 3][Python-badge]][Python-url]
* [![OpenCV][OpenCV-badge]][OpenCV-url]






<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Set up project

First we have to set up a conda environment with all the necessary packages:

* Install conda environment with all requirements
  ```sh
    conda create --name Project1 --file requirements.txt
  ```
* If necessary install this version with pip inside the conda env
  ```sh
    pip install opencv-contrib-python==4.5.5.64
  ```
* Activate conda environment
  ```sh
    conda activate Project1
  ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

## Results

Results can be found in the `result/images` folder.
Here are some example images after applying the pipline and evaluating the results:

| Image 1          | Image 2          |
|-------------------|------------------|
| ![Image 1](results/images/1_result.jpg) | ![Image 2](results/images/2_result.jpg) |



| Image 4          | Image 6          |
|------------------|------------------|
| ![Image 2](results/images/4_result.jpg) | ![Image 3](results/images/6_result.jpg) |



| Image 8          | Image 9          |
|------------------|------------------|
| ![Image 2](results/images/8_result.jpg) | ![Image 3](results/images/9_result.jpg) |

- Green boxes with green dots inside represent true positives: These indicate cases where our pipeline has correctly identified a target, and its prediction aligns with the ground truth.
- Red boxes represent false positives: These are instances where the model has incorrectly identified a target, predicting something that isn't present in the ground truth.
- Blue dots indicate false negatives: These occur when the model misses a target that is present in the ground truth, failing to make a prediction for it.

For more information regarding the evaluation please have a look into our Report about the project.


<!-- CONTACT -->
## Contact

Micha Fauth  - micha.fauth@googlemail.com <br>
Antoni Bennasar Garau - toni.benn.g@gmail.com








<!-- MARKDOWN LINKS & IMAGES -->
[OpenCV-badge]: https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[Python-badge]: https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
