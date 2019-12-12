#!/usr/bin/env python3
'''
This is a client for MPD to generate a random playlist starting from the last
song of the current playlist and iterating using values computed using Bliss.

MPD connection settings are taken from environment variables,
following MPD_HOST and MPD_PORT scheme described in `mpc` man.

You can pass an integer argument to the script to change the length of the
generated playlist (default is to add 20 songs).
'''
import argparse
from itertools import accumulate
import logging
import random
import os
import sqlite3
import socket
import sys
import mpd
import numpy as np

from sklearn.metrics import pairwise_distances

DEFAULT_QUEUE_LENGTH = 20
# 7 hours
GULAG_DURATION = 25200
logging.basicConfig(level=logging.INFO)


if "XDG_DATA_HOME" in os.environ:
    _BLISSIFY_DATA_HOME = os.path.expandvars("$XDG_DATA_HOME/blissify")
else:
    _BLISSIFY_DATA_HOME = os.path.expanduser("~/.local/share/blissify")


class PersistentMPDClient(mpd.MPDClient):
    '''
    From
    https://github.com/schamp/PersistentMPDClient/blob/master/PersistentMPDClient.py
    '''
    def __init__(self, socket=None, host=None, port=None):
        super().__init__()
        self.socket = socket
        self.host = host
        self.port = port

        self.do_connect()
        # get list of available commands from client
        self.command_list = self.commands()

        # commands not to intercept
        self.command_blacklist = ['ping']

        # wrap all valid MPDClient functions
        # in a ping-connection-retry wrapper
        for cmd in self.command_list:
            if cmd not in self.command_blacklist:
                if hasattr(super(PersistentMPDClient, self), cmd):
                    super_fun = (
                        super(PersistentMPDClient, self)
                        .__getattribute__(cmd)
                    )
                    new_fun = self.try_cmd(super_fun)
                    setattr(self, cmd, new_fun)

    # create a wrapper for a function (such as an MPDClient
    # member function) that will verify a connection (and
    # reconnect if necessary) before executing that function.
    # functions wrapped in this way should always succeed
    # (if the server is up)
    # we ping first because we don't want to retry the same
    # function if there's a failure, we want to use the noop
    # to check connectivity
    def try_cmd(self, cmd_fun):
        def fun(*pargs, **kwargs):
            try:
                self.ping()
            except (mpd.ConnectionError, OSError):
                self.do_connect()
            return cmd_fun(*pargs, **kwargs)
        return fun

    def add(self, path):
        super(PersistentMPDClient, self).add(path)
        logging.info('Added song {} to the playlist.'.format(path))

    # needs a name that does not collide with parent connect() function
    def do_connect(self):
        try:
            try:
                self.disconnect()
            # if it's a TCP connection, we'll get a socket error
            # if we try to disconnect when the connection is lost
            except mpd.ConnectionError:
                pass
            # if it's a socket connection, we'll get a BrokenPipeError
            # if we try to disconnect when the connection is lost
            # but we have to retry the disconnect, because we'll get
            # an "Already connected" error if we don't.
            # the second one should succeed.
            except BrokenPipeError:
                try:
                    self.disconnect()
                except Exception:
                    print('Second disconnect failed, yikes.')
            if self.socket:
                self.connect(self.socket, None)
            else:
                self.connect(self.host, self.port)
        except socket.error:
            print("Connection refused.")


def _init():
    # Get MPD connection settings
    try:
        mpd_host = os.environ["MPD_HOST"]
        try:
            mpd_password, mpd_host = mpd_host.split("@")
        except ValueError:
            mpd_password = None
    except KeyError:
        mpd_host = "localhost"
        mpd_password = None
    try:
        mpd_port = os.environ["MPD_PORT"]
    except KeyError:
        mpd_port = 6600

    # Connect to MPD
    client = PersistentMPDClient(host=mpd_host, port=mpd_port)
    if mpd_password is not None:
        client.password(mpd_password)
    # Connect to db
    db_path = os.path.join(_BLISSIFY_DATA_HOME, "db.sqlite3")
    logging.debug("Using DB path: %s." % (db_path,))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute('pragma foreign_keys=ON')
    cur = conn.cursor()

    # Ensure random is not enabled
    status = client.status()
    if int(status["random"]) != 0:
        logging.warning("Random mode is enabled. Are you sure you want it?")

    # Take the last song from current playlist and iterate from it
    playlist = client.playlist()
    if len(playlist) > 0:
        current_song = playlist[-1].replace("file: ", "").rstrip()
    # If current playlist is empty
    else:
        # Add a random song to start with TODO add a random album
        all_songs = [x["file"] for x in client.listall() if "file" in x]
        current_song = random.choice(all_songs)
        client.add(current_song)

    logging.info("Currently played song is %s." % (current_song,))

    # Get current song coordinates
    cur.execute(
        'SELECT id, tempo, amplitude, frequency, attack, filename, '
        'album FROM songs WHERE filename=?',
        (current_song,),
    )
    current_song_coords = cur.fetchone()
    if current_song_coords is None:
        logging.error(
            'Current song {} is not in db. You should update the db.'
            .format((current_song,)),
        )
        client.close()
        client.disconnect()
        sys.exit(1)

    return client, conn, cur, current_song_coords


def make_album_based_playlist(
    first_album_index,
    distance_matrix,
    playlist_length=DEFAULT_QUEUE_LENGTH,
    chained_playlist=False,
):
    return (
        np.argsort(distance_matrix[first_album_index])[1:playlist_length + 1]
    )


def make_song_based_playlist(
    first_song_index,
    distance_matrix,
    playlist_length=DEFAULT_QUEUE_LENGTH,
    chained_playlist=False,
):
    if chained_playlist:
        playlist = []
        current_song_index = first_song_index
        for _ in range(playlist_length + 1):
            # Get the first closest song not already in playlist
            current_song_index = next(
                (
                    s_id
                    for s_id in np.argsort(distance_matrix[current_song_index])
                    if s_id not in playlist
                )
            )
            playlist.append(current_song_index)
        return np.array(playlist[1:])
    return np.argsort(distance_matrix[first_song_index])[1:playlist_length + 1]


def main(opts):
    client, conn, cur, current_song_coords = _init()

    if opts.album_based or opts.gulag_based:
        albums_songs = np.array(
            cur.execute('SELECT id, album FROM songs ORDER BY album')
            .fetchall()
        )
        albums_vector = np.array(
            cur.execute(
                'SELECT album, AVG(tempo), AVG(amplitude), AVG(frequency), '
                'AVG(attack) FROM songs GROUP BY album ORDER BY album'
            ).fetchall()
        )
        distance_matrix = pairwise_distances(albums_vector[:, 1:])
    if opts.album_based:
        # Remove the album name for computing the pairwise distance
        first_album_index = np.where(
            albums_vector[:, 0] == current_song_coords['album']
        )[0][0]
        albums_idx = make_album_based_playlist(
            first_album_index,
            distance_matrix,
            playlist_length=DEFAULT_QUEUE_LENGTH,
            chained_playlist=False,
        )
        playlist_album_names = albums_vector[albums_idx][:, 0]
        playlist_indices = [
            int(song_id) - 1 for song_id, album in albums_songs
            if album in playlist_album_names
        ]
    elif opts.gulag_based:
        client.clear()
        albums_duration = {
            i: sum(map(
                lambda x: float(x.get('duration', 0)),
                client.find('album', album),
            )) for i, album in enumerate(albums_vector[:, 0])
        }
        # Remove the album name for computing the pairwise distance
        first_album_index = np.random.randint(0, len(albums_vector))
        albums_idx = np.argsort(distance_matrix[first_album_index])
        i = next(
            i for i, acc in
            enumerate(accumulate(map(albums_duration.get, albums_idx)))
            if acc > GULAG_DURATION
        )
        playlist_album_names = albums_vector[albums_idx[:i+1]][:, 0]
        playlist_indices = [
            int(song_id) - 1 for song_id, album in albums_songs
            if album in playlist_album_names
        ]
    else:
        force_vectors = np.array(
            cur.execute(
                'SELECT tempo, amplitude, frequency, attack '
                'FROM songs ORDER BY id'
            ).fetchall()[:][:]
        )
        distance_matrix = pairwise_distances(force_vectors)
        playlist_indices = make_song_based_playlist(
            current_song_coords['id'] - 1,
            distance_matrix,
            playlist_length=opts.number_songs,
            chained_playlist=opts.chained_playlist,
        )

    filenames = np.array([
        row['filename']
        for row in cur.execute('SELECT filename FROM songs ORDER BY id')
    ])
    playlist = filenames[playlist_indices]
    for song_path in playlist:
        client.add(song_path)

    conn.close()
    client.close()
    client.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--number-songs',
        help='The number of items to add to the MPD playlist.',
        type=int,
        default=DEFAULT_QUEUE_LENGTH,
    )
    parser.add_argument(
        '-c',
        '--chained-playlist',
        help=(
            'Makes the playlist be generated from one song to the '
            'other closest one, instead of all the songs closer to the '
            'first one.'
        ),
        action='store_true',
        default=False,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-s',
        '--song-based',
        help='Make a playlist based on single songs.',
        action='store_true',
        default=True,
    )
    group.add_argument(
        '-a',
        '--album-based',
        help='Make a playlist based on whole albums.',
        action='store_true',
        default=False,
    )
    group.add_argument(
        '-g',
        '--gulag-based',
        help='Make a 7-hours album playlist for the gulag.',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    main(args)
