/**********************************/
/* Practicos                      */
/* Join de tablas tracks y albums */
/**********************************/

select trackid,name 
from tracks, albums
where tracks.albumid=albums.albumid
and albums.title='Let There Be Rock'

/**********************************/
/* Albums de Iron Maiden          */
/**********************************/

select l.title,t.name 
from artists as a, albums as l, tracks as t 
where a.Name='Iron Maiden' 
and a.artistid=l.artistid 
and l.albumid=t.albumid;

/**********************************/
/* Albums con mas de 25 canciones */
/**********************************/

select l.title, count(*)
from albums a,
tracks t
where a.albumid=t.albumid
group by l.title
having count(*) > 25;

/**********************************/
/*Canciones mas escuchadas        */
/**********************************/

select p.playlistid,t.name,count(1) as cantidad 
from playlists as p, playlist_track as pt,tracks t 
where p.playlistid=pt.playlistid 
and t.trackid=pt.trackid 
group by p.playlistid,t.name 
order by cantidad desc;
