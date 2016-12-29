declare option output:method 'json';

(
let $map := map { 'R': 'red', 'G': 'green', 'B': 'blue' }
return (
  $map?*          (: 1. returns all values; same as: map:keys($map) ! $map(.) :),
  $map?R          (: 2. returns the value associated with the key 'R'; same as: $map('R') :),
  $map?('G','B')  (: 3. returns the values associated with the key 'G' and 'B' :)
),

('A', 'B', 'C') => count(),

for $country in db:open('factbook')//country
where $country/@population > 100000000
let $name := $country/name[1]
for $city in $country//city[population > 1000000]
group by $name
return <country name='{ $name }'>{ $city/name }</country>

)
