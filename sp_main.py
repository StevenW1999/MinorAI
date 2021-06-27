import spotipy
from spotipy.oauth2 import SpotifyOAuth
import main
import spotipyconfig as cfg

scope = """user-library-read, 
        playlist-modify-private, 
        playlist-modify-public, 
        user-read-recently-played"""

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=cfg.client_ID,
        client_secret=cfg.client_SECRET,
        redirect_uri=cfg.redirect_url,
        scope=scope
        )
    )

user_id = sp.current_user()['id']

def get_recently_played(max=10):
    recently_played = sp.current_user_recently_played(limit=max)

    return recently_played

def get_user_playlists(max=10):
    user_saved_playlists = sp.current_user_playlists(limit=max)

    return user_saved_playlists

def get_user_playlists_ids(user_playlists):
    playlist_ids = []
    
    print("all user playlists id:") #debug
    
    for item in user_playlists['items']:
        playlist_id = item['id']
        playlist_ids.append(playlist_id)

    return playlist_ids

def get_user_playlist_names(user_playlists):
    playlist_names = []

    print("all user playlists name:")

    for item in user_playlists['items']:
        playlist_name = item['name']
        playlist_names.append(playlist_name)
    
    return playlist_names

#TODO: finish search implementation
def search(query='', limit=5, type='track'):
    search = sp.search(q='happy', limit=5, type='playlist')

    return search

#Optional
def get_recently_played_tracks(recently_played):
    track_ids = []

    for idx, item in enumerate(recently_played['items']):
        track = item['track']
        track_id = track['id']
        track_ids.append(track_id)
        print(f"{idx, track['artists'][0]['name']} - {track['name']} - {track_id}")

    unique_tracks = list(set(track_ids))

    print(f"unique tracks: {unique_tracks}")

    for idx, item in enumerate(unique_tracks):
        pass

def get_tracks_from_playlist(playlist,max=5):
    pl = sp.playlist(playlist)
    track_ids = []
    for item in pl['tracks']['items']:
        track_ids.append(item['track']['id'])
    tracks = sp.tracks(track_ids)

    print(f"id of tracks: {tracks['tracks'][0]['id']}")

    return tracks

def get_tracks_audio_features_from_playlist(playlist_id):
    tracks = []
    playlist = sp.playlist(playlist_id)
    
    if len(playlist['tracks']['items']) <= 0:
        raise Exception("Make sure the given playlist containts at least 1 track")

    for item in playlist['tracks']['items']:
        track = item['track']
        track_id = track['id']
        # track_name = track['name']
        tracks.append(track_id)
        
    track_features = sp.audio_features(tracks)

    return track_features
        
def get_audio_valences(tracks, valence=0.5):
    
    
    for item in tracks:
        if float(item['valence']) > valence:
            print(item['valence'])

def get_recommendations(artists='', genres='', tracks='', limit=50):
    track_ids = []
    if len(tracks) > 0:
        track_ids = [x['id'] for x in tracks['tracks']]
    recommendations = sp.recommendations(seed_artists=artists, seed_genres=genres, seed_tracks=track_ids, limit=limit)

    return recommendations

def get_recommendations_genres():
    genres = sp.recommendation_genre_seeds()

    return genres

# recently_played = get_recently_played()

# user_playlist = get_user_playlists()

# user_playlist_ids = get_user_playlists_ids(user_playlist)

# get_user_playlist_names(user_playlist)

# search()

# get_recently_played_tracks(recently_played)

# get_audio_valences(get_tracks_audio_features_from_playlist(user_playlist_ids[2]))

# print(get_tracks_from_playlist(user_playlist_ids[2]))

#recommendation = get_recommendations(tracks=get_tracks_from_playlist(user_playlist_ids[2]))

# for item in recommendation['tracks']:
#     print(item['name'])


def get_emotion():
    return main.mood

def valence_level_for_emotion(e):
    if e.lower() == 'sadness':
        valence = 0
    elif e.lower() == 'anger':
        valence = 0.25
    elif e.lower() == 'contempt':
        valence = 0.5
    elif e.lower() == 'happy':
        valence = 0.75

    return valence