## Download Instructions  

1. Download student.zip from this link: https://archive.ics.uci.edu/dataset/320/student+performance
2. Unzip student.zip and place extracted files in /data/datasets

| Variable     | Role    | Type             | Description                                                                                                                     | 
|:------------:|:-------:|:----------------:|---------------------------------------------------------------------------------------------------------------------------------|
| school       | Feature | Categorical      | student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)                                                | 
| sex          | Feature | Binary           | student's sex (binary: 'F' - female or 'M' - male)                                                                              | 
| age          | Feature | Integer          | student's age (numeric: from 15 to 22)                                                                                          | 
| address      | Feature | Categorical      | student's home address type (binary: 'U' - urban or 'R' - rural)                                                                | 
| famsize      | Feature | Categorical      | family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)                                                      | 
| Pstatus      | Feature | Categorical      | parent's cohabitation status (binary: 'T' - living together or 'A' - apart)                                                     | 
| Medu         | Feature | Integer          | mother's  education (numeric: 0 - none,  1 - 4th grade, 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)  |  
| Fedu         | Feature | Integer          | father's  education (numeric: 0 - none,  1 - 4th grade, 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)  | 
| Mjob         | Feature | Categorical      | mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')| 
| Fjob         | Feature | Categorical      | father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')| 
| reason       | Feature | Categorical      | reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')                    | 
| guardian     | Feature | Categorical      | student's guardian (nominal: 'mother', 'father' or 'other')                                                                     | 
| traveltime   | Feature | Integer          | home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)                    | 
| studytime    | Feature | Integer          | weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)                                | 
| failures     | Feature | Integer          | number of past class failures (numeric: n if 1<=n<3, else 4)                                                                    | 
| schoolsup    | Feature | Binary           | extra educational support (binary: yes or no)                                                                                   | 
| famsup       | Feature | Binary           | family educational support (binary: yes or no)                                                                                  | 
| paid         | Feature | Binary           | extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)                                           | 
| activities   | Feature | Binary           | extra-curricular activities (binary: yes or no)                                                                                 | 
| nursery      | Feature | Binary           | attended nursery school (binary: yes or no)                                                                                     | 
| higher       | Feature | Binary           | wants to take higher education (binary: yes or no)                                                                              | 
| internet     | Feature | Binary           | Internet access at home (binary: yes or no)                                                                                     | 
| romantic     | Feature | Binary           | with a romantic relationship (binary: yes or no)                                                                                | 
| famrel       | Feature | Integer          | quality of family relationships (numeric: from 1 - very bad to 5 - excellent)                                                   | 
| freetime     | Feature | Integer          | free time after school (numeric: from 1 - very low to 5 - very high)                                                            | 
| goout        | Feature | Integer          | going out with friends (numeric: from 1 - very low to 5 - very high)                                                            | 
| Dalc         | Feature | Integer          | workday alcohol consumption (numeric: from 1 - very low to 5 - very high)                                                       | 
| Walc         | Feature | Integer          | weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)                                                       | 
| health       | Feature | Integer          | current health status (numeric: from 1 - very bad to 5 - very good)                                                             | 
| absences     | Feature | Integer          | number of school absences (numeric: from 0 to 93)                                                                               | 
| G1           | Target  | Categorical      | first period grade (numeric: from 0 to 20)                                                                                      | 
| G2           | Target  | Categorical      | second period grade (numeric: from 0 to 20)                                                                                     | 
| G3           | Target  | Integer          | final grade (numeric: from 0 to 20, output target)                                                                              | 
: Dataset Information
