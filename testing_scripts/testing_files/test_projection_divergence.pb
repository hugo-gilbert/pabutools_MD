META
key;value
description;Engineered instance to produce different MES outcomes under max vs sum projection. P2 (3-category hub) wins under max due to low max-dimension cost (43.9 vs 100). P3/P4 are cheap per-supporter and selected first under sum; then P1 and P2 compete equally under sum (P1 wins ~50pct of runs when sat_1 > sat_2). Cascade: after P2 is selected nothing else is fundable; after P1 is selected P2 is unfundable. Expected divergence ~40pct of runs.
country;Testland
unit;Divergence City
instance;2025
num_projects;4
num_votes;10
budget;30
vote_type;approval
rule;mes
min_length;1
max_sum_cost;300
PROJECTS
project_id;cost;votes;name;category;target;selected
1;10;1;Green Plaza;green;all;0
2;10;1;Integrated Hub;green,mobility,social;all;0
3;10;1;Cycling Lanes;mobility;cyclists;0
4;10;1;Social Centre;social;residents;0
VOTES
voter_id;vote;age;sex;voting_method
v1;1,2,3,4;25;F;internet
