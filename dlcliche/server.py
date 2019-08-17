import html
import io
import os
import re
import shutil
import sys
import time
import urllib.parse
from http import HTTPStatus
import http.server


class VisibleFileControlHandler(http.server.SimpleHTTPRequestHandler):
    """
    Control visibility of files by regex patterns.
    """

    re_exclude = []
    re_include = []

    @staticmethod
    def set_pattern(exclude_pattern=[], include_pattern=[]):
        VisibleFileControlHandler.re_exclude = [re.compile(p) for p in exclude_pattern]
        VisibleFileControlHandler.re_include = [re.compile(p) for p in include_pattern]

    def _includable(self, text):
        for rex in VisibleFileControlHandler.re_include:
            if rex.search(text):
                return True
        return False

    def _excludable(self, text):
        for rex in VisibleFileControlHandler.re_exclude:
            if rex.search(text):
                if not self._includable(text):
                    return True
        return False

    def list_directory(self, path):
        """
        Mostly copied from https://github.com/python/cpython/blob/master/Lib/http/server.py
        """
        try:
            list = os.listdir(path)
        except OSError:
            self.send_error(
                HTTPStatus.NOT_FOUND,
                "No permission to list directory")
            return None

        list.sort(key=lambda a: a.lower())

        ## *** Customization starts here ***
        # Exclude files
        excludable = [l for l in list if self._excludable(l)]
        list = [l for l in list if l not in excludable]
        #print(list)
        ## *** Customization ends here ***

        r = []
        try:
            displaypath = urllib.parse.unquote(self.path,
                                               errors='surrogatepass')
        except UnicodeDecodeError:
            displaypath = urllib.parse.unquote(path)
        displaypath = html.escape(displaypath, quote=False)
        enc = sys.getfilesystemencoding()
        title = 'Directory listing for %s' % displaypath
        r.append('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" '
                 '"http://www.w3.org/TR/html4/strict.dtd">')
        r.append('<html>\n<head>')
        r.append('<meta http-equiv="Content-Type" '
                 'content="text/html; charset=%s">' % enc)
        r.append('<title>%s</title>\n</head>' % title)
        r.append('<body>\n<h1>%s</h1>' % title)
        r.append('<hr>\n<ul>')
        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            r.append('<li><a href="%s">%s</a></li>'
                    % (urllib.parse.quote(linkname,
                                          errors='surrogatepass'),
                       html.escape(displayname, quote=False)))
        r.append('</ul>\n<hr>\n</body>\n</html>\n')
        encoded = '\n'.join(r).encode(enc, 'surrogateescape')
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f

def file_visible_controlled_server(server_address=("", 8000), exclude_pattern=[], include_pattern=[]):
    """
    Simple http file server that controls file/folder visibility.
    """
    VisibleFileControlHandler.set_pattern(exclude_pattern, include_pattern)
    httpd = http.server.HTTPServer(server_address, VisibleFileControlHandler)
    httpd.serve_forever()


if __name__ == '__main__':
    # Run example file server that doesn't show files usually unneeded.
    file_visible_controlled_server(exclude_pattern=[r'^\.', r'\.bak', r'^~'])
