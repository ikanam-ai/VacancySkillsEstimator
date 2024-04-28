import fitz

class ResumeSkillsFinder:
    """
    Класс для поиска навыков в резюме.

    Атрибуты:
        skills (list): Список навыков для поиска в резюме.
    """
    def __init__(self) -> None:
        """
        Инициализирует объект класса ResumeSkillsFinder, загружая навыки из файла.

        Параметры:
            skills_path (str): Путь к файлу с навыками.
        """
        self.skills = None
   # @staticmethod
   # def _load_skills(skills_path: str) -> list:
   #     with open(skills_path, 'r', encoding='utf-8') as file:
   #         skills = file.read().split()
   #     return skills

    @staticmethod
    def _read_pdf(path: str) -> str:
        with fitz.open(path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text

    def find_skills_in_resume(self, resume_path: str) -> list:
        """
        Ищет навыки в тексте резюме.

        Параметры:
            resume_path (str): Путь к PDF-файлу с резюме.

        Возвращает:
            found_skills (list): Список найденных навыков.
        """
        resume_text = self._read_pdf(resume_path)
        found_skills = [skill for skill in self.skills if skill.lower() in resume_text.lower()]
        return found_skills
