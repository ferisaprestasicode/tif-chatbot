from config import get_connection, logger

from datetime import datetime
from collections import defaultdict

import re



def escape_special_characters(text):
    # Escape karakter newline, tab, dan backslash
    text = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    # Escape tanda kutip ganda
    text = re.sub(r'"', r'\\"', text)
    return text

def insert_history_chat(session_id, email, user, bot, category):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        # Query untuk menyisipkan data ke tabel chat_history
        insert_query = """
        INSERT INTO chat_history (session_id, email, user_chat, bot_chat, timestamp, category)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        # Menyusun data yang akan disisipkan
        timestamp = datetime.now()  # Waktu saat data dimasukkan
        cursor.execute(insert_query, (session_id, email, user, bot, timestamp, category))
        
        # Commit transaksi
        connection.commit()
        logger.info("Data berhasil dimasukkan!")
        print("Data berhasil dimasukkan!")
    except Exception as e:
        logger.info(f"Terjadi kesalahan insert: {e}")
        print(f"Terjadi kesalahan insert: {e}")
        connection.rollback()

def fetch_chat_history_by_email(email, category):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        # Query untuk mengambil 10 percakapan terakhir berdasarkan email
        select_query = """
        SELECT session_id, email, user_chat, bot_chat, timestamp, category
        FROM chat_history
        WHERE email = %s
        AND category = %s
        ORDER BY timestamp DESC
        LIMIT 50;
        """
        
        # Eksekusi query dengan parameter email
        cursor.execute(select_query, (email,category,))
        
        # Ambil hasil query
        results = cursor.fetchall()
        
        # Jika ada hasil, tampilkan
        if results:
            # Mengelompokkan data berdasarkan session_id
            chat_history = defaultdict(list)
            for row in results:
                chat_history[row[0]].append({
                    "email": row[1],
                    "user": row[2],
                    "bot": escape_special_characters(row[3]),
                    "timestamp": row[4].strftime("%Y-%m-%d %H:%M:%S"),
                    "category": row[5]
                })
            
            # Mengubah dictionary menjadi list of session histories
            grouped_history = [
                {"session_id": session_id, "chat": chats} 
                for session_id, chats in chat_history.items()
            ]
            
            return grouped_history
        else:
            return []  # Tidak ada data ditemukan
    
    except Exception as e:
        logger.info(f"Terjadi kesalahan fetching: {e}")
    finally:
        cursor.close()
        connection.close()

def delete_session_chat(session_id):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        # Query untuk menyisipkan data ke tabel chat_history
        query = """
        DELETE FROM chat_history
        WHERE session_id = %s
        """
        
        # Menyusun data yang akan disisipkan
        timestamp = datetime.now()  # Waktu saat data dimasukkan
        cursor.execute(query, (session_id,))
        
        # Commit transaksi
        connection.commit()
        logger.info("Data chat berhasil dihapus!")
    except Exception as e:
        logger.info(f"Terjadi kesalahan delete: {e}")
        connection.rollback()

def save_message(user_session, role, content):
    try:
        # Membuka koneksi ke database
        connection = get_connection()
        cursor = connection.cursor()
        
        # Query untuk menyimpan pesan ke database
        insert_query = """
        INSERT INTO conversation_history (user_session, role, content, timestamp) 
        VALUES (%s, %s, %s, %s)
        """
        
        # Menjalankan query dengan parameter
        cursor.execute(insert_query, (user_session, role, content, datetime.utcnow()))
        
        # Commit untuk menyimpan perubahan
        connection.commit()
        
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat menyimpan pesan: {e}", exc_info=True)
    finally:
        # Menutup cursor dan koneksi
        cursor.close()
        connection.close()


def fetch_conversation_history(session_id):
    try:
        # Membuka koneksi ke database
        connection = get_connection()
        cursor = connection.cursor()
        
        # Query untuk mengambil percakapan berdasarkan session_id
        query = """
        SELECT role, content FROM conversation_history WHERE user_session = %s ORDER BY timestamp DESC LIMIT 6
        """
        
        # Eksekusi query dengan parameter session_id
        cursor.execute(query, (session_id,))
        
        # Ambil hasil query
        rows = cursor.fetchall()
        
        # Mengembalikan hasil percakapan dalam format yang diinginkan
        return [{"role": row[0], "content": row[1].lower()} for row in rows]
        
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat mengambil pesan: {e}", exc_info=True)
    finally:
        # Menutup cursor dan koneksi
        cursor.close()
        connection.close()

