import sp_main as s
import random

emotion = s.get_emotion()

emotion_valence = s.valence_level_for_emotion(emotion)

search = s.search(query=emotion)

print(s.get_recommendations_genres())
genres = input('Please specify a genre from the list above. If more than one is desired (max 5), please separate them with a space: ')
genres = genres.split()

recommendations = s.get_recommendations(genres=genres)
recommendations_tracks_id = [track['id'] for track in recommendations['tracks']]

audio_features = s.sp.audio_features(recommendations_tracks_id)

id_valence_dict = {} 
for track in audio_features:
    id_valence_dict.update({track['id']: track['valence']})


new_tracks = []
for item, item2 in id_valence_dict.items():
    if float(item2) > emotion_valence and float(item2) <= emotion_valence + 0.25:
        print(f"Track id: {item}, valence: {item2}")
        new_tracks.append(item)


user_id = s.user_id
add_playlist = s.sp.user_playlist_create(user=user_id, name=f"Emotion based playlist {random.randint(0, 1000)}")
user_playlist = s.sp.current_user_playlists(limit=5)
latest_playlist_id = user_playlist['items'][0]['id']
add_tracks = s.sp.user_playlist_add_tracks(user=user_id, playlist_id=latest_playlist_id, tracks=new_tracks)
print('succesfully added tracks to playlist!')