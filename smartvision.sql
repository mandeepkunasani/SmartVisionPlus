CREATE DATABASE IF NOT EXISTS smartvision;
USE smartvision;

-- 🧑‍🏫 Faculty Table
CREATE TABLE IF NOT EXISTS faculty (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100) UNIQUE,
  password VARCHAR(100),
  department VARCHAR(50),
  class VARCHAR(10)
);

-- 🎓 Students Table
CREATE TABLE IF NOT EXISTS students (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  reg_no VARCHAR(50) UNIQUE,
  class_name VARCHAR(50),
  face_encoding LONGBLOB,
  face_image VARCHAR(255)
);

-- 📅 Attendance Table
CREATE TABLE IF NOT EXISTS attendance (
  id INT AUTO_INCREMENT PRIMARY KEY,
  student_id INT,
  class_name VARCHAR(50),
  date DATE,
  time TIME,
  status VARCHAR(10),
  FOREIGN KEY (student_id) REFERENCES students(id)
);