//! Minimal terminal handling on top of libc: raw-ish input mode, alternate
//! screen, window size, non-blocking key polling, quit-on-signal.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

static QUIT: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    QUIT.store(true, Ordering::SeqCst);
}

/// Install handlers so SIGINT/SIGTERM/SIGHUP end the loop cleanly
/// (and the terminal gets restored).
pub fn install_signal_handlers() {
    for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        unsafe {
            libc::signal(
                sig,
                on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t,
            );
        }
    }
}

pub fn quit_requested() -> bool {
    QUIT.load(Ordering::SeqCst)
}

/// Terminal size as (cols, rows). Falls back to $COLUMNS/$LINES, then 100×30.
pub fn size() -> (usize, usize) {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) == 0
            && ws.ws_col > 0
            && ws.ws_row > 0
        {
            return (ws.ws_col as usize, ws.ws_row as usize);
        }
    }
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse().ok());
    (env("COLUMNS").unwrap_or(100), env("LINES").unwrap_or(30))
}

pub fn stdout_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDOUT_FILENO) == 1 }
}

pub fn stdin_is_tty() -> bool {
    unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
}

/// Puts the terminal into cbreak mode (no echo, no line buffering, no signal
/// keys) on the alternate screen with a hidden cursor. Restores everything on
/// drop — including on panic.
pub struct Screen {
    saved: Option<libc::termios>,
}

impl Screen {
    pub fn enter() -> io::Result<Self> {
        let saved = unsafe {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut t) == 0 {
                let saved = t;
                t.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
                t.c_cc[libc::VMIN] = 1;
                t.c_cc[libc::VTIME] = 0;
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t);
                Some(saved)
            } else {
                None // stdin not a terminal: keys just won't work
            }
        };
        let mut out = io::stdout().lock();
        out.write_all(b"\x1b[?1049h\x1b[?25l\x1b[2J")?;
        out.flush()?;
        Ok(Screen { saved })
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        let mut out = io::stdout().lock();
        let _ = out.write_all(b"\x1b[0m\x1b[?25h\x1b[?1049l");
        let _ = out.flush();
        if let Some(t) = self.saved {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSADRAIN, &t);
            }
        }
    }
}

/// Wait up to `timeout_ms` for input and return the bytes read (possibly none).
pub fn poll_keys(timeout_ms: i32, buf: &mut [u8]) -> &[u8] {
    let mut pfd = libc::pollfd {
        fd: libc::STDIN_FILENO,
        events: libc::POLLIN,
        revents: 0,
    };
    let ready = unsafe { libc::poll(&mut pfd, 1, timeout_ms.max(0)) };
    if ready <= 0 || pfd.revents & libc::POLLIN == 0 {
        return &[];
    }
    let n = unsafe { libc::read(libc::STDIN_FILENO, buf.as_mut_ptr().cast(), buf.len()) };
    if n <= 0 {
        &[]
    } else {
        &buf[..n as usize]
    }
}
