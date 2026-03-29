"""Intent: Scrum Card Operations

Shows current card details from the scrum board.
"""

import os
import sqlite3
from parsely_dip.engine.registry import intent


def get_db_path():
    """Get path to bibliotech.db."""
    project_dir = os.getenv('CLAUDE_PROJECT_DIR', '.')
    return os.path.join(project_dir, 'db', 'bibliotech.db')


def query(sql, params=(), single=False):
    """Simple query helper."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    if single:
        return rows[0] if rows else None
    return rows


@intent('show_current_card')
def show_current_card():
    """Show the current card in Doing."""
    # Get cards in Doing (list_name = 'Doing') on bibliotech board
    cards = query("""
        SELECT c.card_id, c.card_title, c.card_descr, c.card_points,
               l.list_name, b.board_name
        FROM scrum_cards c
        JOIN scrum_lists l ON c.list_id = l.list_id
        JOIN scrum_boards b ON l.board_id = b.board_id
        WHERE l.list_name = 'Doing' AND c.card_is_archived = 0
              AND b.board_id = 4
        ORDER BY c.card_order
    """)

    if not cards:
        return "No cards in Doing."

    lines = []
    for card in cards:
        lines.append(f"#{card['card_id']} [{card['card_points']}pts] {card['card_title']}")
        lines.append(f"  Board: {card['board_name']}")
        if card['card_descr']:
            lines.append(f"  {card['card_descr']}")

        # Get tasks
        tasks = query("""
            SELECT task_text, task_done
            FROM scrum_tasks
            WHERE card_id = ?
            ORDER BY task_id
        """, (card['card_id'],))

        if tasks:
            done = sum(1 for t in tasks if t['task_done'])
            lines.append(f"  Tasks: {done}/{len(tasks)}")
            for t in tasks:
                mark = 'x' if t['task_done'] else ' '
                lines.append(f"    [{mark}] {t['task_text']}")

        lines.append("")

    return "\n".join(lines).strip()


@intent('read_current_card')
def read_current_card():
    """Read current cards — same data, LLM summarizes."""
    return show_current_card()
