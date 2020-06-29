# Protien-Processor

Project outline

This project was a group effort between the computer science department and the biochemistry department at ENMU.

One of the great goals of modern biochemistry is to be able to predict the properties of a protein based on the sequence of amino acids in the protein.

Allow me to explain the complexity of the problem as it was pitched to me. There are 20 common amino acids and a number of more uncommon ones which can be chained together in any sequence.
So a protien of length 3 may look like A-G-C with 'A' 'G' and 'C' each representing a different amino acid.
With our current understanding, we don't have a way of meaningfully predicting how the protien's properties would change if it was instead A-G-G without directly testing that protien.
That leaves us with about 20^3 or 8,000 possible protiens that have to be sythensized and tested in order to find out the properties of all protiens of length 3.
Now consider that hemoglobin, one of the most commonly known proteins contains 574 amino acids, and you can begin to see the scale of the problem.
This research aims to find ways to more accurately predict the properties of proteins by collecting large amounts of uniform data and creating a structure to facilitate collaboration across many labs the world over.



The goal of this project was to create a webpage that could be accessed by other universities to uniformly automate the processing of protein data.
This is meant to be a broad design to be built upon by future computer science students, but I'll go over what we accomplished with this initial build here.

We managed to design and build a website that was functional and easy to use on the back of the Flask platform. It's lightweight and functional, for our proof of concept, the server ran on a raspberry pi kept in the department head's office.
It featured a backend app, also running in python, that was able to process and normalize large data files produced by a robotic testing station.
Once the initial processing was complete the uploader would be shown all the final data curves and remove any erroneous results caused by the robotic station.
The new data set would then be processed, and sent to the uploader as a download as well as archived in the database.

Some samples of the uploaded data as well as the final results are available in the /tmp folder of this github.

Thanks for checking out our work!
