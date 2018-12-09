# Bohemian-Alligators
<img src="logo.jpg"  height="250" width="250" align="middle"> 

<h1>Motivation</h1>
This project was created as a part of the course  Deep Learning in Practice with Python and LUA (Budapest University of Technology and Economics). We enrolled this course with no prior knowledge or experience in the field of deep learning, therefore our sole purpose was to gain a deeper understanding of the theoretical concepts while working with them in practice. We approached the assignment with a curious, experimental view.

<h4>Team members:</h4>
- Csenge Kilián<br>
- Beáta Csilla Kovács<br>

<h1>Description of the Project</h1>
This project's aim is to create a deep neural network trained with the UC Berkeley's dataset which is capable of semantic segmentation of cityscapes. Our main goal is to be able to differenciate vehicles and pedestrians.

<h1>Directory Structure</h1>

<pre><code>
dataset/
  catid_annot
  class_color
  ram_images
  categories.csv
model_weights/
  model.36-0.9417.hdf5
Documentation.doc
README.md
evaluation.ipynb
network.ipynb
</code></pre>

old mappaban a segnet-es dolgok

<h1>The Dataset</h1>
The dataset can be downloaded on the following link: http://bdd-data.berkeley.edu/ <br>
The data folder contains a reduced set of US Berkerley's dataset.

<h1>Training and Testing</h1>


<h4>UC Berkeley's Standard Copyright and Disclaimer Notice:</h4>

Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and
distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a
signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, for commercial licensing opportunities.

Contributors of BDD Data, University of California, Berkeley. 
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
