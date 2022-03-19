# This Python file uses the following encoding: utf-8

import re
from bs4 import BeautifulSoup

# SourceP: https://ar.wikipedia.org/wiki/%D9%82%D8%A7%D8%A6%D9%85%D8%A9_%D8%A3%D8%B4%D8%AE%D8%A7%D8%B5_%D8%B0%D9%83%D8%B1%D9%88%D8%A7_%D9%81%D9%8A_%D8%A7%D9%84%D9%83%D8%AA%D8%A7%D8%A8_%D8%A7%D9%84%D9%85%D9%82%D8%AF%D8%B3_%D9%88%D8%A7%D9%84%D9%82%D8%B1%D8%A2%D9%86
html_table = """<table class="wikitable">

<tbody><tr>
<th>الاسم في الكتاب المقدس (النسخة العربية)
</th>
<th>الاسم في القرآن الكريم
</th>
<th>ملاحظات
</th></tr>
<tr>
<td><a href="/wiki/%D8%A2%D8%AF%D9%85" title="آدم">آدم</a>
</td>
<td>آدم
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A5%D8%A8%D8%B1%D8%A7%D9%87%D9%8A%D9%85" title="إبراهيم">إبراهيم</a>
</td>
<td>إبراهيم
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A5%D9%8A%D9%84%D9%8A%D8%A7" title="إيليا">إيليا</a>
</td>
<td><a href="/wiki/%D8%A5%D9%84%D9%8A%D8%A7%D8%B3" title="إلياس">إلياس</a>
</td>
<td>
</td></tr>
<tr>
<td>إليشع
</td>
<td><a href="/wiki/%D8%A7%D9%84%D9%8A%D8%B3%D8%B9" title="اليسع">اليسع</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%AA%D9%84%D8%A7%D9%85%D9%8A%D8%B0_%D8%A7%D9%84%D9%85%D8%B3%D9%8A%D8%AD" title="تلاميذ المسيح">تلاميذ المسيح</a>
</td>
<td><a href="/wiki/%D8%A7%D9%84%D8%AD%D9%88%D8%A7%D8%B1%D9%8A%D9%88%D9%86" class="mw-redirect" title="الحواريون">الحواريون</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%AF%D8%A7%D9%88%D9%88%D8%AF" class="mw-redirect" title="داوود">داوود</a>
</td>
<td>داوود
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%B9%D8%A7%D8%A8%D8%B1" title="عابر">عابر</a>
</td>
<td><a href="/wiki/%D9%87%D9%88%D8%AF" title="هود">هود</a>
</td>
<td>يرى البعض أن عابر في النصوص اليهودية هو <a href="/wiki/%D9%87%D9%88%D8%AF" title="هود">هود</a> في النصوص الإسلامية <sup id="cite_ref-1" class="reference"><a href="#cite_note-1">[1]</a></sup><sup id="cite_ref-2" class="reference"><a href="#cite_note-2">[2]</a></sup>
</td></tr>
<tr>
<td><a href="/wiki/%D8%B9%D9%85%D8%B1%D8%A7%D9%85" title="عمرام">عمرام</a>
</td>
<td><a href="/wiki/%D8%B9%D9%85%D8%B1%D8%A7%D9%86" class="mw-redirect mw-disambig" title="عمران">عمران</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%82%D8%A7%D9%8A%D9%8A%D9%86" class="mw-redirect" title="قايين">قايين</a>
</td>
<td><a href="/wiki/%D9%82%D8%A7%D8%A8%D9%8A%D9%84" class="mw-redirect" title="قابيل">قابيل</a>
</td>
<td>لم يذكر القرآن الكريم اسمي قابيل صراحةً، بل ورد ذكرهما تحت اسم "ابني آدم".
</td></tr>
<tr>
<td><a href="/wiki/%D9%87%D8%A7%D8%A8%D9%8A%D9%84" class="mw-redirect" title="هابيل">هابيل</a>
</td>
<td>هابيل
</td>
<td>لم يذكر القرآن الكريم اسم هابيل صراحةً، بل ورد ذكرهما تحت اسم "ابني آدم".
</td></tr>
<tr>
<td><a href="/wiki/%D9%87%D8%A7%D8%B1%D9%88%D9%86" title="هارون">هارون</a>
</td>
<td>هارون
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A3%D8%AE%D9%86%D9%88%D8%AE" class="mw-redirect" title="أخنوخ">أخنوخ</a>
</td>
<td><a href="/wiki/%D8%A5%D8%AF%D8%B1%D9%8A%D8%B3" title="إدريس">إدريس</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%AD%D8%B2%D9%82%D9%8A%D8%A7%D9%84" title="حزقيال">حزقيال</a>
</td>
<td><a href="/wiki/%D8%B0%D9%88_%D8%A7%D9%84%D9%83%D9%81%D9%84" title="ذو الكفل">ذو الكفل</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%B9%D8%B2%D8%B1%D8%A7" title="عزرا">عزرا</a>
</td>
<td><a href="/wiki/%D8%B9%D8%B2%D9%8A%D8%B1" title="عزير">عزير</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%AC%D8%A8%D8%B1%D8%A7%D8%A6%D9%8A%D9%84" title="جبرائيل">جبرائيل</a>
</td>
<td><a href="/wiki/%D8%AC%D8%A8%D8%B1%D9%8A%D9%84" title="جبريل">جبريل</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%AC%D9%88%D8%AC_%D9%88%D9%85%D8%A7%D8%AC%D9%88%D8%AC" class="mw-redirect" title="جوج وماجوج">جوج وماجوج</a>
</td>
<td><a href="/wiki/%D9%8A%D8%A3%D8%AC%D9%88%D8%AC_%D9%88%D9%85%D8%A3%D8%AC%D9%88%D8%AC" title="يأجوج ومأجوج">يأجوج ومأجوج</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%AC%D9%84%D9%8A%D8%A7%D8%AA" class="mw-redirect" title="جليات">جليات</a>
</td>
<td><a href="/wiki/%D8%AC%D8%A7%D9%84%D9%88%D8%AA" title="جالوت">جالوت</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%87%D8%A7%D9%85%D8%A7%D9%86" title="هامان">هامان</a>
</td>
<td><a href="/wiki/%D9%87%D8%A7%D9%85%D8%A7%D9%86" title="هامان">هامان</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A5%D8%B3%D8%AD%D8%A7%D9%82" title="إسحاق">إسحاق</a>
</td>
<td><a href="/wiki/%D8%A5%D8%B3%D8%AD%D8%A7%D9%82" title="إسحاق">إسحاق</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A5%D8%B3%D9%85%D8%A7%D8%B9%D9%8A%D9%84" title="إسماعيل">إسماعيل</a>
</td>
<td><a href="/wiki/%D8%A5%D8%B3%D9%85%D8%A7%D8%B9%D9%8A%D9%84" title="إسماعيل">إسماعيل</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%8A%D8%B9%D9%82%D9%88%D8%A8" title="يعقوب">يعقوب</a>
</td>
<td><a href="/wiki/%D9%8A%D8%B9%D9%82%D9%88%D8%A8" title="يعقوب">يعقوب</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%8A%D8%AB%D8%B1%D9%88%D9%86" class="mw-redirect" title="يثرون">يثرون</a>، <a href="/wiki/%D8%B1%D8%B9%D9%88%D8%A6%D9%8A%D9%84" title="رعوئيل">رعوئيل</a>، <a href="/wiki/%D8%AD%D9%88%D8%A8%D8%A7%D8%A8" class="mw-redirect" title="حوباب">حوباب</a>
</td>
<td><a href="/wiki/%D8%B4%D8%B9%D9%8A%D8%A8" title="شعيب">شعيب</a>
</td>
<td>هذا الترادف غير مؤكد، رغم أن الثلاثة كانوا من أهل مدين.
</td></tr>
<tr>
<td><a href="/wiki/%D9%8A%D8%B3%D9%88%D8%B9" title="يسوع">يسوع</a>
</td>
<td><a href="/wiki/%D8%B9%D9%8A%D8%B3%D9%89" class="mw-redirect" title="عيسى">عيسى</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%8A%D9%87%D9%88%D9%8A%D8%A7%D9%82%D9%8A%D9%85" title="يهوياقيم">يهوياقيم</a> أو <a href="/wiki/%D9%87%D8%A7%D9%84%D9%8A" class="mw-redirect mw-disambig" title="هالي">هالي</a>
</td>
<td>عمران (أبو <a href="/wiki/%D9%85%D8%B1%D9%8A%D9%85_%D8%A8%D9%86%D8%AA_%D8%B9%D9%85%D8%B1%D8%A7%D9%86" title="مريم بنت عمران">مريم</a>)
</td>
<td>لا توجد علاقة <a href="/wiki/%D8%AA%D8%A3%D8%B5%D9%8A%D9%84" class="mw-redirect" title="تأصيل">تأصيلية</a> بين هذه الأسماء.
</td></tr>
<tr>
<td><a href="/wiki/%D8%A3%D9%8A%D9%88%D8%A8" title="أيوب">أيوب</a>
</td>
<td><a href="/wiki/%D8%A3%D9%8A%D9%88%D8%A8" title="أيوب">أيوب</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%8A%D9%88%D8%AD%D9%86%D8%A7_%D8%A7%D9%84%D9%85%D8%B9%D9%85%D8%AF%D8%A7%D9%86" title="يوحنا المعمدان">يوحنا المعمدان</a>
</td>
<td><a href="/wiki/%D9%8A%D8%AD%D9%8A%D9%89_%D8%A8%D9%86_%D8%B2%D9%83%D8%B1%D9%8A%D8%A7" title="يحيى بن زكريا">يحيى بن زكريا</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%8A%D9%88%D9%86%D8%B3" title="يونس">يونان</a>
</td>
<td><a href="/wiki/%D9%8A%D9%88%D9%86%D8%B3" title="يونس">يونس</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%8A%D9%88%D8%B3%D9%81" title="يوسف">يوسف</a>
</td>
<td><a href="/wiki/%D9%8A%D9%88%D8%B3%D9%81_%D9%81%D9%8A_%D8%A7%D9%84%D8%A5%D8%B3%D9%84%D8%A7%D9%85" title="يوسف في الإسلام">يوسف</a>
</td>
<td>
</td></tr>
<tr>
<td>إخوة يوسف
</td>
<td>إخوة يوسف
</td>
<td>يورد الكتاب المقدس أسماءهم، بينما لم تذكر في القرآن الكريم.
</td></tr>
<tr>
<td><a href="/w/index.php?title=%D9%82%D9%88%D8%B1%D8%AD&amp;action=edit&amp;redlink=1" class="new" title="قورح (الصفحة غير موجودة)">قورح</a>
</td>
<td><a href="/wiki/%D9%82%D8%A7%D8%B1%D9%88%D9%86" title="قارون">قارون</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%84%D9%88%D8%B7" title="لوط">لوط</a>
</td>
<td><a href="/wiki/%D9%84%D9%88%D8%B7" title="لوط">لوط</a>
</td>
<td>
</td></tr>
<tr>
<td>امرأة لوط
</td>
<td>امرأة لوط
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%85%D8%B1%D9%8A%D9%85_%D8%A7%D9%84%D8%B9%D8%B0%D8%B1%D8%A7%D8%A1" title="مريم العذراء">مريم العذراء</a>
</td>
<td><a href="/wiki/%D9%85%D8%B1%D9%8A%D9%85_%D8%A8%D9%86%D8%AA_%D8%B9%D9%85%D8%B1%D8%A7%D9%86" title="مريم بنت عمران">مريم بنت عمران</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%85%D8%B1%D9%8A%D9%85_(%D8%A3%D8%AE%D8%AA_%D9%85%D9%88%D8%B3%D9%89)" title="مريم (أخت موسى)">مريم (أخت موسى)</a>
</td>
<td>أخت موسى
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%B1%D8%A6%D9%8A%D8%B3_%D8%A7%D9%84%D9%85%D9%84%D8%A7%D8%A6%D9%83%D8%A9_%D9%85%D9%8A%D8%AE%D8%A7%D8%A6%D9%8A%D9%84" class="mw-redirect" title="رئيس الملائكة ميخائيل">رئيس الملائكة ميخائيل</a>
</td>
<td>ميكائيل
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%85%D9%88%D8%B3%D9%89" title="موسى">موسى</a>
</td>
<td><a href="/wiki/%D9%85%D9%88%D8%B3%D9%89_%D9%81%D9%8A_%D8%A7%D9%84%D8%A5%D8%B3%D9%84%D8%A7%D9%85" title="موسى في الإسلام">موسى</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%86%D9%88%D8%AD" title="نوح">نوح</a>
</td>
<td><a href="/wiki/%D9%86%D9%88%D8%AD" title="نوح">نوح</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D9%81%D8%B1%D8%B9%D9%88%D9%86" title="فرعون">فرعون</a>
</td>
<td><a href="/wiki/%D9%81%D8%B1%D8%B9%D9%88%D9%86" title="فرعون">فرعون</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A8%D9%88%D8%AA%D9%8A%D9%81%D8%A7%D8%B1" title="بوتيفار">بوتيفار</a>
</td>
<td>العزيز
</td>
<td>
</td></tr>
<tr>
<td>امرأة العزيز
</td>
<td>امرأة العزيز، <a href="/wiki/%D8%B2%D9%84%D9%8A%D8%AE%D8%A9" title="زليخة">زليخة</a>
</td>
<td>مصدر الاسم الإسلامي هو حكايات متواترة.
</td></tr>
<tr>
<td><a href="/wiki/%D9%85%D9%84%D9%83%D8%A9_%D8%B3%D8%A8%D8%A3" class="mw-redirect" title="ملكة سبأ">ملكة سبأ</a>
</td>
<td><a href="/wiki/%D9%85%D9%84%D9%83%D8%A9_%D8%B3%D8%A8%D8%A3" class="mw-redirect" title="ملكة سبأ">ملكة سبأ</a>، <a href="/wiki/%D8%A8%D9%84%D9%82%D9%8A%D8%B3" title="بلقيس">بلقيس</a>
</td>
<td>لم يرد اسم بلقيس في القرآن الكريم، بل جاء من القصص العربية القديمة.
</td></tr>
<tr>
<td><a href="/wiki/%D8%B4%D8%A7%D9%88%D9%84_%D8%A7%D9%84%D9%85%D9%84%D9%83" class="mw-redirect" title="شاول الملك">شاول الملك</a>
</td>
<td><a href="/wiki/%D8%B7%D8%A7%D9%84%D9%88%D8%AA" class="mw-redirect" title="طالوت">طالوت</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A7%D9%84%D8%B4%D9%8A%D8%B7%D8%A7%D9%86" class="mw-redirect" title="الشيطان">الشيطان</a>
</td>
<td><a href="/wiki/%D8%A5%D8%A8%D9%84%D9%8A%D8%B3" title="إبليس">إبليس</a> أو <a href="/wiki/%D8%A7%D9%84%D8%B4%D9%8A%D8%B7%D8%A7%D9%86" class="mw-redirect" title="الشيطان">الشيطان</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%B3%D8%A7%D9%85" title="سام">سام</a> <a href="/wiki/%D8%AD%D8%A7%D9%85" title="حام">وحام</a> <a href="/wiki/%D9%8A%D8%A7%D9%81%D8%AB" title="يافث">ويافث</a>
</td>
<td>أبناء نوح
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%A7%D9%84%D9%85%D9%84%D9%83_%D8%B3%D9%84%D9%8A%D9%85%D8%A7%D9%86" class="mw-redirect" title="الملك سليمان">الملك سليمان</a>
</td>
<td><a href="/wiki/%D8%B3%D9%84%D9%8A%D9%85%D8%A7%D9%86" title="سليمان">سليمان</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%AA%D8%A7%D8%B1%D8%AD" title="تارح">تارح</a>
</td>
<td><a href="/wiki/%D8%A2%D8%B2%D8%B1" class="mw-redirect" title="آزر">آزر</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%B2%D9%83%D8%B1%D9%8A%D8%A7" title="زكريا">زكريا</a>
</td>
<td><a href="/wiki/%D8%B2%D9%83%D8%B1%D9%8A%D8%A7" title="زكريا">زكريا</a>
</td>
<td>
</td></tr>
<tr>
<td><a href="/wiki/%D8%B2%D9%85%D8%B1%D9%8A_%D8%A8%D9%86_%D8%B3%D8%A7%D9%84%D9%88" class="mw-redirect" title="زمري بن سالو">زمري بن سالو</a>
</td>
<td><a href="/wiki/%D8%A7%D9%84%D8%B3%D8%A7%D9%85%D8%B1%D9%8A" title="السامري">السامري</a>
</td>
<td>
</td></tr></tbody></table>"""

soup = BeautifulSoup(html_table, "html.parser")

# Parse second column in each row skipping the header
persons = [
    person.find_all("td")[1].get_text().strip() for person in soup.find_all("tr")[1:]
]

# Remove text between ()
persons = [re.sub(r"[(].*[)]", "", p).strip() for p in persons]

# Split multiple variants of the same name
persons = sum(
    [
        [p]
        if not any([mark in p for mark in ["أو", "،"]])
        else p.split("أو")
        if "أو" in p
        else p.split("،")
        for p in persons
    ],
    [],
)
# Remove extra whitespaces
persons = [p.strip() for p in persons]
persons = sorted(set(persons))

# Source: https://ar.wikipedia.org/wiki/%D8%B4%D8%AE%D8%B5%D9%8A%D8%A7%D8%AA_%D8%B0%D9%83%D8%B1%D8%AA_%D9%81%D9%8A_%D8%A7%D9%84%D9%82%D8%B1%D8%A2%D9%86
lists_of_persons = [
    """<ul><li><a href="/wiki/%D9%85%D9%88%D8%B3%D9%89" title="موسى">موسى</a> (136 مرة)</li>
<li><a href="/wiki/%D8%A5%D8%A8%D8%B1%D8%A7%D9%87%D9%8A%D9%85" title="إبراهيم">إبراهيم</a> (69 مرة)</li>
<li><a href="/wiki/%D9%86%D9%88%D8%AD" title="نوح">نوح</a> (43 مرة)</li>
<li><a href="/wiki/%D9%84%D9%88%D8%B7" title="لوط">لوط</a> (27 مرة)</li>
<li><a href="/wiki/%D9%8A%D9%88%D8%B3%D9%81" title="يوسف">يوسف</a> (27 مرة)</li>
<li><a href="/wiki/%D8%A2%D8%AF%D9%85" title="آدم">آدم</a> (25 مرة)</li>
<li><a href="/wiki/%D9%87%D9%88%D8%AF" title="هود">هود</a> (7 مرات)</li>
<li><a href="/wiki/%D8%B9%D9%8A%D8%B3%D9%89" class="mw-redirect" title="عيسى">عيسى</a> (25 مرة)</li>
<li><a href="/wiki/%D9%87%D8%A7%D8%B1%D9%88%D9%86" title="هارون">هارون</a> (20 مرة)</li>
<li><a href="/wiki/%D8%A5%D8%B3%D8%AD%D8%A7%D9%82" title="إسحاق">إسحاق</a> (17 مرة)</li>
<li><a href="/wiki/%D8%B3%D9%84%D9%8A%D9%85%D8%A7%D9%86" title="سليمان">سليمان</a> (17 مرة)</li>
<li><a href="/wiki/%D8%AF%D8%A7%D9%88%D8%AF" title="داود">داود</a> (16 مرة)</li>
<li><a href="/wiki/%D9%8A%D8%B9%D9%82%D9%88%D8%A8" title="يعقوب">يعقوب</a> (16 مرة) + باسم إسرائيل (43) مرة</li>
<li><a href="/wiki/%D8%A5%D8%B3%D9%85%D8%A7%D8%B9%D9%8A%D9%84" title="إسماعيل">إسماعيل</a> (12 مرة)</li>
<li><a href="/wiki/%D8%B4%D8%B9%D9%8A%D8%A8" title="شعيب">شعيب</a> (11 مرات)</li>
<li><a href="/wiki/%D8%B5%D8%A7%D9%84%D8%AD" title="صالح">صالح</a> (9 مرات)</li>
<li><a href="/wiki/%D8%B2%D9%83%D8%B1%D9%8A%D8%A7" title="زكريا">زكريا</a> (7 مرات)</li>
<li><a href="/wiki/%D9%85%D8%AD%D9%85%D8%AF" title="محمد">محمد</a> (4 مرات)+ مرة واحدة باسم (<a href="/wiki/%D8%A3%D8%AD%D9%85%D8%AF_(%D8%A7%D8%B3%D9%85)" title="أحمد (اسم)">أحمد</a>)</li>
<li><a href="/wiki/%D8%A3%D9%8A%D9%88%D8%A8" title="أيوب">أيوب</a> (4 مرات)</li>
<li><a href="/wiki/%D9%8A%D9%88%D9%86%D8%B3" title="يونس">يونس</a> (4 مرات)+ مرة واحدة (ذو النون)</li>
<li><a href="/wiki/%D9%8A%D8%AD%D9%8A%D9%89_%D8%A8%D9%86_%D8%B2%D9%83%D8%B1%D9%8A%D8%A7" title="يحيى بن زكريا">يحيى</a> (5 مرة)</li>
<li><a href="/wiki/%D8%A7%D9%84%D9%8A%D8%B3%D8%B9" title="اليسع">اليسع</a> (مرتان)</li>
<li><a href="/wiki/%D8%B0%D9%88_%D8%A7%D9%84%D9%83%D9%81%D9%84" title="ذو الكفل">ذو الكفل</a> (مرتان)</li>
<li><a href="/wiki/%D8%A5%D9%84%D9%8A%D8%A7%D8%B3" title="إلياس">إلياس</a> (مرتان) + مرة واحدة (الياسين)</li>
<li><a href="/wiki/%D8%A5%D8%AF%D8%B1%D9%8A%D8%B3" title="إدريس">إدريس</a> (مرتان)</li></ul>""",
    """<ul><li><a href="/wiki/%D8%A2%D8%B2%D8%B1" class="mw-redirect" title="آزر">آزر</a> (مرة واحدة) (سورة الأنعام)</li>
<li><a href="/wiki/%D9%81%D8%B1%D8%B9%D9%88%D9%86" title="فرعون">فرعون</a> (74 مرة) لم يكن اسما ولكن كان لقبًا يطلق على ملوك مصر.</li>
<li><a href="/wiki/%D9%82%D9%88%D9%85_%D8%AA%D8%A8%D8%B9" title="قوم تبع">تبع</a> (مرتان) وكان يطلق على ملوك اليمن.</li>
<li><a href="/wiki/%D9%8A%D9%87%D9%88%D9%8A%D8%A7%D9%82%D9%8A%D9%85" title="يهوياقيم">عمران</a> (3 مرات)</li>
<li><a href="/wiki/%D9%85%D8%B1%D9%8A%D9%85_%D8%A8%D9%86%D8%AA_%D8%B9%D9%85%D8%B1%D8%A7%D9%86" title="مريم بنت عمران">مريم</a> (34 مرة)</li>
<li><a href="/wiki/%D9%87%D8%A7%D9%85%D8%A7%D9%86" title="هامان">هامان</a> (6 مرات) ويقال ان هامان ليس إسما ولكن وظيفة للوزير الخاص بالبناء</li>
<li><a href="/wiki/%D9%82%D8%A7%D8%B1%D9%88%D9%86" title="قارون">قارون</a> (4 مرات)</li>
<li><a href="/wiki/%D8%B0%D9%88_%D8%A7%D9%84%D9%82%D8%B1%D9%86%D9%8A%D9%86" title="ذو القرنين">ذو القرنين</a> (3 مرات)</li>
<li><a href="/wiki/%D8%A7%D9%84%D8%B3%D8%A7%D9%85%D8%B1%D9%8A" title="السامري">السامري</a> (3 مرات)</li>
<li><a href="/wiki/%D8%AC%D8%A7%D9%84%D9%88%D8%AA" title="جالوت">جالوت</a> (3 مرات)</li>
<li><a href="/wiki/%D9%84%D9%82%D9%85%D8%A7%D9%86_%D8%A7%D9%84%D8%AD%D9%83%D9%8A%D9%85" title="لقمان الحكيم">لقمان</a> (مرتان)</li>
<li><a href="/wiki/%D8%B7%D8%A7%D9%84%D9%88%D8%AA" class="mw-redirect" title="طالوت">طالوت</a> (مرتان)</li>
<li><a href="/wiki/%D8%B9%D8%B2%D9%8A%D8%B1" title="عزير">عزير</a> (مرة ومرة أخرى ذكرت قصته بغير اسمه <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">سورة البقرة</a> آية 259)</li>
<li><a href="/wiki/%D8%A3%D8%A8%D9%88_%D9%84%D9%87%D8%A8" title="أبو لهب">أبو لهب</a> (مرة؛ <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%B3%D8%AF" title="سورة المسد">سورة المسد</a>)</li>
<li><a href="/wiki/%D8%B2%D9%8A%D8%AF_%D8%A8%D9%86_%D8%AD%D8%A7%D8%B1%D8%AB%D8%A9" title="زيد بن حارثة">زيد بن حارثة</a> (مرة) (<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%AD%D8%B2%D8%A7%D8%A8" title="سورة الأحزاب">سورة الأحزاب</a>: 37)</li></ul>""",
    """<ul><li><a href="/wiki/%D8%AC%D8%A8%D8%B1%D8%A7%D8%A6%D9%8A%D9%84" title="جبرائيل">جبرائيل</a> (3 مرات)</li>
<li><a href="/wiki/%D9%85%D9%8A%D9%83%D8%A7%D8%A6%D9%8A%D9%84" title="ميكائيل">ميكائيل</a> (مرة)</li>
<li><a href="/wiki/%D9%87%D8%A7%D8%B1%D9%88%D8%AA_%D9%88%D9%85%D8%A7%D8%B1%D9%88%D8%AA" title="هاروت وماروت">هاروت</a> (مرة)</li>
<li><a href="/wiki/%D9%87%D8%A7%D8%B1%D9%88%D8%AA_%D9%88%D9%85%D8%A7%D8%B1%D9%88%D8%AA" title="هاروت وماروت">ماروت</a> (مرة)</li>
<li><a href="/wiki/%D9%85%D8%A7%D9%84%D9%83_%D8%AE%D8%A7%D8%B2%D9%86_%D8%A7%D9%84%D9%86%D8%A7%D8%B1" class="mw-redirect" title="مالك خازن النار">مالك خازن النار</a> (مرة)</li></ul>""",
    """<ul><li><a href="/wiki/%D8%A7%D9%84%D9%84%D8%A7%D8%AA" title="اللات">اللات</a></li>
<li><a href="/wiki/%D9%85%D9%86%D8%A7%D8%A9" title="مناة">مناة</a></li>
<li><a href="/wiki/%D8%A7%D9%84%D8%B9%D8%B2%D9%89" title="العزى">العزى</a></li>
<li><a href="/wiki/%D8%B3%D9%88%D8%A7%D8%B9" title="سواع">سواع</a></li>
<li><a href="/wiki/%D9%8A%D8%BA%D9%88%D8%AB" title="يغوث">يغوث</a></li>
<li><a href="/wiki/%D9%86%D8%B3%D8%B1" title="نسر">نسر</a></li>
<li><a href="/wiki/%D8%A8%D8%B9%D9%84" title="بعل">بعل</a></li>
<li><a href="/wiki/%D9%8A%D8%B9%D9%88%D9%82" title="يعوق">يعوق</a></li></ul>""",
    """<ul><li><a href="/wiki/%D8%A5%D8%A8%D9%84%D9%8A%D8%B3" title="إبليس">إبليس</a> (11 مرة)</li></ul>""",
]

for list_of_persons in lists_of_persons:
    soup = BeautifulSoup(list_of_persons, "html.parser")

    for li in soup.find_all("li"):
        persons.append(li.find_all("a")[0].text)


def get_augmented_value(person):
    if re.search(r"^ذو\s", person):
        return re.sub("^ذو", "ذي", person)
    if re.search(r"^أبو\s", person):
        return re.sub("^أبو", "أبي", person)
    if re.search(r"^ال.*ون$", person):
        return re.sub("ون$", "ين", person)
    return None


persons = (
    persons
    + [get_augmented_value(p) for p in persons if get_augmented_value(p)]
    + ["عيسى ابن مريم", "قوم تبع"]
    + ["ثمود", "إرم"]
    + ["الأسباط"]
    + ["يهود", "نصاري"]
    + ["ملائكة", "شياطين"]
    # + ["ثمود", "عاد", "إرم"]
)
persons = [p for p in persons if p != "تبع"]
persons = sorted(set([p.strip() for p in persons]))

import os

os.makedirs("data", exist_ok=True)
with open("data/persons.txt", "w") as f:
    f.write("\n".join(persons))

list_of_animals = """<ol><li><a href="/wiki/%D8%A3%D8%B3%D8%AF" title="أسد">الأسد</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%AF%D8%AB%D8%B1" title="سورة المدثر">سورة المدثر</a> ( 51 )&nbsp;: <style data-mw-deduplicate="TemplateStyles:r56526893">.mw-parser-output .quran-cite{font-size:80%;vertical-align:5%}.mw-parser-output .quran{font-family:"Amiri Quran",Amiri,"Sakkal Majalla","Noto Naskh Arabic","Traditional Arabic","Microsoft Uighur",Tahoma,"Segoe UI",Arial!important}</style><span class="quran">﴿كَأَنَّهُمْ حُمُرٌ مُسْتَنْفِرَةٌ <style data-mw-deduplicate="TemplateStyles:r56889424">.mw-parser-output .end-of-aya{display:inline-block;position:relative;width:20px;font-family:"KFGQPC Uthman Taha Naskh","Amiri Quran",Amiri,Calibri,"Noto Naskh Arabic","Sakkal Majalla","Traditional Arabic","Microsoft Uighur",Tahoma,"Segoe UI",Arial}.mw-parser-output .aya-num{left:0;top:50%;width:20px;position:absolute;font-size:9px;font-weight:500;text-align:center;transform:translateY(-43%)}</style><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٥٠</span></span> فَرَّتْ مِنْ قَسْوَرَةٍ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٥١</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%AF%D8%AB%D8%B1" title="سورة المدثر">المدثر</a>:50–51]</small></span>  «قسورة أي الأسد<sup id="cite_ref-2" class="reference"><a href="#cite_note-2">[2]</a></sup>»</li>
<li><a href="/wiki/%D8%A8%D8%BA%D9%84" title="بغل">البغل</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">سورة النحل</a> ( 8 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَالْخَيْلَ وَالْبِغَالَ وَالْحَمِيرَ لِتَرْكَبُوهَا وَزِينَةً وَيَخْلُقُ مَا لَا تَعْلَمُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٨</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">النحل</a>:8]</small></span></li>
<li><a href="/wiki/%D8%A8%D9%82%D8%B1%D8%A9" class="mw-redirect" title="بقرة">البقرة</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">سورة البقرة</a> ( 67 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَإِذْ قَالَ مُوسَى لِقَوْمِهِ إِنَّ اللَّهَ يَأْمُرُكُمْ أَنْ تَذْبَحُوا بَقَرَةً قَالُوا أَتَتَّخِذُنَا هُزُوًا قَالَ أَعُوذُ بِاللَّهِ أَنْ أَكُونَ مِنَ الْجَاهِلِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٦٧</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">البقرة</a>:67]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">سورة الأنعام</a> ( 146 )&nbsp;:  <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَعَلَى الَّذِينَ هَادُوا حَرَّمْنَا كُلَّ ذِي ظُفُرٍ وَمِنَ الْبَقَرِ وَالْغَنَمِ حَرَّمْنَا عَلَيْهِمْ شُحُومَهُمَا إِلَّا مَا حَمَلَتْ ظُهُورُهُمَا أَوِ الْحَوَايَا أَوْ مَا اخْتَلَطَ بِعَظْمٍ ذَلِكَ جَزَيْنَاهُمْ بِبَغْيِهِمْ وَإِنَّا لَصَادِقُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤٦</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">الأنعام</a>:146]</small></span></li>
<li><a href="/wiki/%D8%A8%D8%B9%D9%88%D8%B6%D9%8A%D8%A7%D8%AA" title="بعوضيات">البعوضة</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">سورة البقرة</a> ( 26 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿إِنَّ اللَّهَ لَا يَسْتَحْيِي أَنْ يَضْرِبَ مَثَلًا مَا بَعُوضَةً فَمَا فَوْقَهَا فَأَمَّا الَّذِينَ آمَنُوا فَيَعْلَمُونَ أَنَّهُ الْحَقُّ مِنْ رَبِّهِمْ وَأَمَّا الَّذِينَ كَفَرُوا فَيَقُولُونَ مَاذَا أَرَادَ اللَّهُ بِهَذَا مَثَلًا يُضِلُّ بِهِ كَثِيرًا وَيَهْدِي بِهِ كَثِيرًا وَمَا يُضِلُّ بِهِ إِلَّا الْفَاسِقِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٢٦</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">البقرة</a>:26]</small></span></li>
<li><a href="/wiki/%D8%AB%D8%B9%D8%A8%D8%A7%D9%86" title="ثعبان">الثعبان</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">سورة الأعراف</a> ( 107 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿قَالَ إِنْ كُنْتَ جِئْتَ بِآيَةٍ فَأْتِ بِهَا إِنْ كُنْتَ مِنَ الصَّادِقِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٠٦</span></span> فَأَلْقَى عَصَاهُ فَإِذَا هِيَ ثُعْبَانٌ مُبِينٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٠٧</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">الأعراف</a>:106–107]</small></span>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%B4%D8%B9%D8%B1%D8%A7%D8%A1" title="سورة الشعراء">سورة الشعراء</a> ( 32 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿قَالَ فَأْتِ بِهِ إِنْ كُنْتَ مِنَ الصَّادِقِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٣١</span></span> فَأَلْقَى عَصَاهُ فَإِذَا هِيَ ثُعْبَانٌ مُبِينٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٣٢</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%B4%D8%B9%D8%B1%D8%A7%D8%A1" title="سورة الشعراء">الشعراء</a>:31–32]</small></span></li>
<li><a href="/wiki/%D8%AC%D8%B1%D8%A7%D8%AF%D8%A9_(%D8%AD%D8%B4%D8%B1%D8%A9)" class="mw-redirect" title="جرادة (حشرة)">الجراد</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">سورة الأعراف</a> ( 133 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿فَأَرْسَلْنَا عَلَيْهِمُ الطُّوفَانَ وَالْجَرَادَ وَالْقُمَّلَ وَالضَّفَادِعَ وَالدَّمَ آيَاتٍ مُفَصَّلَاتٍ فَاسْتَكْبَرُوا وَكَانُوا قَوْمًا مُجْرِمِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٣٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">الأعراف</a>:133]</small></span>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%82%D9%85%D8%B1" title="سورة القمر">سورة القمر</a> ( 7 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿خُشَّعًا أَبْصَارُهُمْ يَخْرُجُونَ مِنَ الْأَجْدَاثِ كَأَنَّهُمْ جَرَادٌ مُنْتَشِرٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٧</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%82%D9%85%D8%B1" title="سورة القمر">القمر</a>:7]</small></span></li>
<li><a href="/wiki/%D8%AC%D9%85%D9%84" title="جمل">الجمل</a> وذكر بصيغ عديدة وهي&nbsp;: <a href="/wiki/%D8%AC%D9%85%D9%84" title="جمل">الجمل</a>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">سورة الأعراف</a> ( 40 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿إِنَّ الَّذِينَ كَذَّبُوا بِآيَاتِنَا وَاسْتَكْبَرُوا عَنْهَا لَا تُفَتَّحُ لَهُمْ أَبْوَابُ السَّمَاءِ وَلَا يَدْخُلُونَ الْجَنَّةَ حَتَّى يَلِجَ الْجَمَلُ فِي سَمِّ الْخِيَاطِ وَكَذَلِكَ نَجْزِي الْمُجْرِمِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٤٠</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">الأعراف</a>:40]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%B1%D8%B3%D9%84%D8%A7%D8%AA" title="سورة المرسلات">سورة المرسلات</a> ( 33 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿إِنَّهَا تَرْمِي بِشَرَرٍ كَالْقَصْرِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٣٢</span></span> كَأَنَّهُ جِمَالَةٌ صُفْرٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٣٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%B1%D8%B3%D9%84%D8%A7%D8%AA" title="سورة المرسلات">المرسلات</a>:32–33]</small></span>  <a href="/wiki/%D8%AC%D9%85%D9%84" title="جمل">الابل</a>    - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">سورة الأنعام</a> ( 144 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَمِنَ الْإِبِلِ اثْنَيْنِ وَمِنَ الْبَقَرِ اثْنَيْنِ قُلْ آلذَّكَرَيْنِ حَرَّمَ أَمِ الْأُنْثَيَيْنِ أَمَّا اشْتَمَلَتْ عَلَيْهِ أَرْحَامُ الْأُنْثَيَيْنِ أَمْ كُنْتُمْ شُهَدَاءَ إِذْ وَصَّاكُمُ اللَّهُ بِهَذَا فَمَنْ أَظْلَمُ مِمَّنِ افْتَرَى عَلَى اللَّهِ كَذِبًا لِيُضِلَّ النَّاسَ بِغَيْرِ عِلْمٍ إِنَّ اللَّهَ لَا يَهْدِي الْقَوْمَ الظَّالِمِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤٤</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">الأنعام</a>:144]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%BA%D8%A7%D8%B4%D9%8A%D8%A9" title="سورة الغاشية">سورة الغاشية</a> ( 17 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿أَفَلَا يَنْظُرُونَ إِلَى الْإِبِلِ كَيْفَ خُلِقَتْ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٧</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%BA%D8%A7%D8%B4%D9%8A%D8%A9" title="سورة الغاشية">الغاشية</a>:17]</small></span>  وجاءت بصيغة <b>الناقة</b> و<b>البدن</b> و<b>البعير</b> .</li>
<li><a href="/wiki/%D8%AD%D9%85%D8%A7%D8%B1" title="حمار">الحمار</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%AC%D9%85%D8%B9%D8%A9" title="سورة الجمعة">سورة الجمعة</a> ( 5 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿مَثَلُ الَّذِينَ حُمِّلُوا التَّوْرَاةَ ثُمَّ لَمْ يَحْمِلُوهَا كَمَثَلِ الْحِمَارِ يَحْمِلُ أَسْفَارًا بِئْسَ مَثَلُ الْقَوْمِ الَّذِينَ كَذَّبُوا بِآيَاتِ اللَّهِ وَاللَّهُ لَا يَهْدِي الْقَوْمَ الظَّالِمِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٥</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%AC%D9%85%D8%B9%D8%A9" title="سورة الجمعة">الجمعة</a>:5]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">سورة النحل</a> ( 8 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَالْخَيْلَ وَالْبِغَالَ وَالْحَمِيرَ لِتَرْكَبُوهَا وَزِينَةً وَيَخْلُقُ مَا لَا تَعْلَمُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٨</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">النحل</a>:8]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D9%84%D9%82%D9%85%D8%A7%D9%86" title="سورة لقمان">سورة لقمان</a> ( 19 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَاقْصِدْ فِي مَشْيِكَ وَاغْضُضْ مِنْ صَوْتِكَ إِنَّ أَنْكَرَ الْأَصْوَاتِ لَصَوْتُ الْحَمِيرِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٩</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D9%84%D9%82%D9%85%D8%A7%D9%86" title="سورة لقمان">لقمان</a>:19]</small></span></li>
<li><a href="/wiki/%D8%AD%D9%8A%D8%AA%D8%A7%D9%86%D9%8A%D8%A7%D8%AA" title="حيتانيات">الحوت</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%B5%D8%A7%D9%81%D8%A7%D8%AA" title="سورة الصافات">سورة الصافات</a> ( 142 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿فَالْتَقَمَهُ الْحُوتُ وَهُوَ مُلِيمٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤٢</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%B5%D8%A7%D9%81%D8%A7%D8%AA" title="سورة الصافات">الصافات</a>:142]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%82%D9%84%D9%85" title="سورة القلم">سورة القلم</a> ( 48 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿فَاصْبِرْ لِحُكْمِ رَبِّكَ وَلَا تَكُنْ كَصَاحِبِ الْحُوتِ إِذْ نَادَى وَهُوَ مَكْظُومٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٤٨</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%82%D9%84%D9%85" title="سورة القلم">القلم</a>:48]</small></span></li>
<li><a href="/wiki/%D8%AD%D8%B5%D8%A7%D9%86" class="mw-redirect" title="حصان">الخيل</a>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A2%D9%84_%D8%B9%D9%85%D8%B1%D8%A7%D9%86" title="سورة آل عمران">سورة آل عمران</a> ( 14 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿زُيِّنَ لِلنَّاسِ حُبُّ الشَّهَوَاتِ مِنَ النِّسَاءِ وَالْبَنِينَ وَالْقَنَاطِيرِ الْمُقَنْطَرَةِ مِنَ الذَّهَبِ وَالْفِضَّةِ وَالْخَيْلِ الْمُسَوَّمَةِ وَالْأَنْعَامِ وَالْحَرْثِ ذَلِكَ مَتَاعُ الْحَيَاةِ الدُّنْيَا وَاللَّهُ عِنْدَهُ حُسْنُ الْمَآبِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A2%D9%84_%D8%B9%D9%85%D8%B1%D8%A7%D9%86" title="سورة آل عمران">آل عمران</a>:14]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">سورة النحل</a> ( 8 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَالْخَيْلَ وَالْبِغَالَ وَالْحَمِيرَ لِتَرْكَبُوهَا وَزِينَةً وَيَخْلُقُ مَا لَا تَعْلَمُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٨</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">النحل</a>:8]</small></span></li>
<li><a href="/wiki/%D8%AE%D9%86%D8%B2%D9%8A%D8%B1" title="خنزير">الخنزير</a> - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">سورة البقرة</a> ( 173 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿إِنَّمَا حَرَّمَ عَلَيْكُمُ الْمَيْتَةَ وَالدَّمَ وَلَحْمَ الْخِنْزِيرِ وَمَا أُهِلَّ بِهِ لِغَيْرِ اللَّهِ فَمَنِ اضْطُرَّ غَيْرَ بَاغٍ وَلَا عَادٍ فَلَا إِثْمَ عَلَيْهِ إِنَّ اللَّهَ غَفُورٌ رَحِيمٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٧٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">البقرة</a>:173]</small></span>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">سورة المائدة</a> ( 60 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿قُلْ هَلْ أُنَبِّئُكُمْ بِشَرٍّ مِنْ ذَلِكَ مَثُوبَةً عِنْدَ اللَّهِ مَنْ لَعَنَهُ اللَّهُ وَغَضِبَ عَلَيْهِ وَجَعَلَ مِنْهُمُ الْقِرَدَةَ وَالْخَنَازِيرَ وَعَبَدَ الطَّاغُوتَ أُولَئِكَ شَرٌّ مَكَانًا وَأَضَلُّ عَنْ سَوَاءِ السَّبِيلِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٦٠</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">المائدة</a>:60]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">سورة الأنعام</a> ( 145 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿قُلْ لَا أَجِدُ فِي مَا أُوحِيَ إِلَيَّ مُحَرَّمًا عَلَى طَاعِمٍ يَطْعَمُهُ إِلَّا أَنْ يَكُونَ مَيْتَةً أَوْ دَمًا مَسْفُوحًا أَوْ لَحْمَ خِنْزِيرٍ فَإِنَّهُ رِجْسٌ أَوْ فِسْقًا أُهِلَّ لِغَيْرِ اللَّهِ بِهِ فَمَنِ اضْطُرَّ غَيْرَ بَاغٍ وَلَا عَادٍ فَإِنَّ رَبَّكَ غَفُورٌ رَحِيمٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤٥</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">الأنعام</a>:145]</small></span></li>
<li><a href="/wiki/%D8%B0%D9%88%D8%A7%D8%AA_%D8%A7%D9%84%D8%AC%D9%86%D8%A7%D8%AD%D9%8A%D9%86" title="ذوات الجناحين">الذباب</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%AD%D8%AC" title="سورة الحج">سورة الحج</a> ( 73 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿يَا أَيُّهَا النَّاسُ ضُرِبَ مَثَلٌ فَاسْتَمِعُوا لَهُ إِنَّ الَّذِينَ تَدْعُونَ مِنْ دُونِ اللَّهِ لَنْ يَخْلُقُوا ذُبَابًا وَلَوِ اجْتَمَعُوا لَهُ وَإِنْ يَسْلُبْهُمُ الذُّبَابُ شَيْئًا لَا يَسْتَنْقِذُوهُ مِنْهُ ضَعُفَ الطَّالِبُ وَالْمَطْلُوبُ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٧٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%AD%D8%AC" title="سورة الحج">الحج</a>:73]</small></span></li>
<li><a href="/wiki/%D8%B0%D8%A6%D8%A8_%D8%B1%D9%85%D8%A7%D8%AF%D9%8A" title="ذئب رمادي">الذئب</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D9%8A%D9%88%D8%B3%D9%81" title="سورة يوسف">سورة يوسف</a> ( 13-14-17 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿قَالَ إِنِّي لَيَحْزُنُنِي أَنْ تَذْهَبُوا بِهِ وَأَخَافُ أَنْ يَأْكُلَهُ الذِّئْبُ وَأَنْتُمْ عَنْهُ غَافِلُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٣</span></span> قَالُوا لَئِنْ أَكَلَهُ الذِّئْبُ وَنَحْنُ عُصْبَةٌ إِنَّا إِذًا لَخَاسِرُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D9%8A%D9%88%D8%B3%D9%81" title="سورة يوسف">يوسف</a>:13–14]</small></span><link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿قَالُوا يَا أَبَانَا إِنَّا ذَهَبْنَا نَسْتَبِقُ وَتَرَكْنَا يُوسُفَ عِنْدَ مَتَاعِنَا فَأَكَلَهُ الذِّئْبُ وَمَا أَنْتَ بِمُؤْمِنٍ لَنَا وَلَوْ كُنَّا صَادِقِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٧</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D9%8A%D9%88%D8%B3%D9%81" title="سورة يوسف">يوسف</a>:17]</small></span></li>
<li><a href="/wiki/%D8%B3%D9%85%D8%A7%D9%86" title="سمان">طائر السلوى</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">سورة البقرة</a> ( 57 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَظَلَّلْنَا عَلَيْكُمُ الْغَمَامَ وَأَنْزَلْنَا عَلَيْكُمُ الْمَنَّ وَالسَّلْوَى كُلُوا مِنْ طَيِّبَاتِ مَا رَزَقْنَاكُمْ وَمَا ظَلَمُونَا وَلَكِنْ كَانُوا أَنْفُسَهُمْ يَظْلِمُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٥٧</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">البقرة</a>:57]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%B7%D9%87" title="سورة طه">سورة طه</a> ( 80 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿يَا بَنِي إِسْرَائِيلَ قَدْ أَنْجَيْنَاكُمْ مِنْ عَدُوِّكُمْ وَوَاعَدْنَاكُمْ جَانِبَ الطُّورِ الْأَيْمَنَ وَنَزَّلْنَا عَلَيْكُمُ الْمَنَّ وَالسَّلْوَى <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٨٠</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%B7%D9%87" title="سورة طه">طه</a>:80]</small></span></li>
<li><a href="/wiki/%D8%B6%D8%A3%D9%86" class="mw-redirect" title="ضأن">الضأن</a> - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">سورة الأنعام</a> ( 143 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿ثَمَانِيَةَ أَزْوَاجٍ مِنَ الضَّأْنِ اثْنَيْنِ وَمِنَ الْمَعْزِ اثْنَيْنِ قُلْ آلذَّكَرَيْنِ حَرَّمَ أَمِ الْأُنْثَيَيْنِ أَمَّا اشْتَمَلَتْ عَلَيْهِ أَرْحَامُ الْأُنْثَيَيْنِ نَبِّئُونِي بِعِلْمٍ إِنْ كُنْتُمْ صَادِقِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">الأنعام</a>:143]</small></span></li>
<li><a href="/wiki/%D8%A7%D9%84%D8%B6%D9%81%D8%AF%D8%B9" class="mw-redirect" title="الضفدع">الضفدع</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">سورة الأعراف</a> ( 133 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿فَأَرْسَلْنَا عَلَيْهِمُ الطُّوفَانَ وَالْجَرَادَ وَالْقُمَّلَ وَالضَّفَادِعَ وَالدَّمَ آيَاتٍ مُفَصَّلَاتٍ فَاسْتَكْبَرُوا وَكَانُوا قَوْمًا مُجْرِمِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٣٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">الأعراف</a>:133]</small></span></li>
<li><a href="/wiki/%D8%A8%D9%82%D8%B1%D8%A9" class="mw-redirect" title="بقرة">العجل</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%B3%D8%A7%D8%A1" title="سورة النساء">سورة النساء</a> ( 153 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿يَسْأَلُكَ أَهْلُ الْكِتَابِ أَنْ تُنَزِّلَ عَلَيْهِمْ كِتَابًا مِنَ السَّمَاءِ فَقَدْ سَأَلُوا مُوسَى أَكْبَرَ مِنْ ذَلِكَ فَقَالُوا أَرِنَا اللَّهَ جَهْرَةً فَأَخَذَتْهُمُ الصَّاعِقَةُ بِظُلْمِهِمْ ثُمَّ اتَّخَذُوا الْعِجْلَ مِنْ بَعْدِ مَا جَاءَتْهُمُ الْبَيِّنَاتُ فَعَفَوْنَا عَنْ ذَلِكَ وَآتَيْنَا مُوسَى سُلْطَانًا مُبِينًا <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٥٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%B3%D8%A7%D8%A1" title="سورة النساء">النساء</a>:153]</small></span>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D9%87%D9%88%D8%AF" title="سورة هود">سورة هود</a> 69 <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَلَقَدْ جَاءَتْ رُسُلُنَا إِبْرَاهِيمَ بِالْبُشْرَى قَالُوا سَلَامًا قَالَ سَلَامٌ فَمَا لَبِثَ أَنْ جَاءَ بِعِجْلٍ حَنِيذٍ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٦٩</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D9%87%D9%88%D8%AF" title="سورة هود">هود</a>:69]</small></span> - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">سورة الاعراف</a> ( 152 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿إِنَّ الَّذِينَ اتَّخَذُوا الْعِجْلَ سَيَنَالُهُمْ غَضَبٌ مِنْ رَبِّهِمْ وَذِلَّةٌ فِي الْحَيَاةِ الدُّنْيَا وَكَذَلِكَ نَجْزِي الْمُفْتَرِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٥٢</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">الأعراف</a>:152]</small></span> - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">سورة البقرة</a> وذكر في ثلاث مواضع ( 51 - 54 - 92 - 93 ) .</li>
<li><a href="/wiki/%D8%B9%D9%86%D9%83%D8%A8%D9%88%D8%AA" title="عنكبوت">العنكبوت</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%B9%D9%86%D9%83%D8%A8%D9%88%D8%AA" title="سورة العنكبوت">سورة العنكبوت</a> ( 41 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿مَثَلُ الَّذِينَ اتَّخَذُوا مِنْ دُونِ اللَّهِ أَوْلِيَاءَ كَمَثَلِ الْعَنْكَبُوتِ اتَّخَذَتْ بَيْتًا وَإِنَّ أَوْهَنَ الْبُيُوتِ لَبَيْتُ الْعَنْكَبُوتِ لَوْ كَانُوا يَعْلَمُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٤١</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%B9%D9%86%D9%83%D8%A8%D9%88%D8%AA" title="سورة العنكبوت">العنكبوت</a>:41]</small></span></li>
<li><a href="/wiki/%D9%81%D8%B1%D8%A7%D8%B4%D8%A9" title="فراشة">الفراشة</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%82%D8%A7%D8%B1%D8%B9%D8%A9" title="سورة القارعة">سورة القارعة</a> ( 4 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿يَوْمَ يَكُونُ النَّاسُ كَالْفَرَاشِ الْمَبْثُوثِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٤</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%82%D8%A7%D8%B1%D8%B9%D8%A9" title="سورة القارعة">القارعة</a>:4]</small></span></li>
<li><a href="/wiki/%D9%82%D8%B1%D8%AF" class="mw-redirect" title="قرد">القرد</a>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">سورة البقرة</a> ( 65 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَلَقَدْ عَلِمْتُمُ الَّذِينَ اعْتَدَوْا مِنْكُمْ فِي السَّبْتِ فَقُلْنَا لَهُمْ كُونُوا قِرَدَةً خَاسِئِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٦٥</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A8%D9%82%D8%B1%D8%A9" title="سورة البقرة">البقرة</a>:65]</small></span>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">سورة المائدة</a> ( 60 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿قُلْ هَلْ أُنَبِّئُكُمْ بِشَرٍّ مِنْ ذَلِكَ مَثُوبَةً عِنْدَ اللَّهِ مَنْ لَعَنَهُ اللَّهُ وَغَضِبَ عَلَيْهِ وَجَعَلَ مِنْهُمُ الْقِرَدَةَ وَالْخَنَازِيرَ وَعَبَدَ الطَّاغُوتَ أُولَئِكَ شَرٌّ مَكَانًا وَأَضَلُّ عَنْ سَوَاءِ السَّبِيلِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٦٠</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">المائدة</a>:60]</small></span></li>
<li><a href="/wiki/%D9%82%D9%85%D9%84" class="mw-redirect" title="قمل">قمل</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">سورة الأعراف</a> ( 133 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿فَأَرْسَلْنَا عَلَيْهِمُ الطُّوفَانَ وَالْجَرَادَ وَالْقُمَّلَ وَالضَّفَادِعَ وَالدَّمَ آيَاتٍ مُفَصَّلَاتٍ فَاسْتَكْبَرُوا وَكَانُوا قَوْمًا مُجْرِمِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٣٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">الأعراف</a>:133]</small></span></li>
<li><a href="/wiki/%D9%83%D9%84%D8%A8" title="كلب">الكلب</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">سورة الأعراف</a> ( 77 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَلَوْ شِئْنَا لَرَفَعْنَاهُ بِهَا وَلَكِنَّهُ أَخْلَدَ إِلَى الْأَرْضِ وَاتَّبَعَ هَوَاهُ فَمَثَلُهُ كَمَثَلِ الْكَلْبِ إِنْ تَحْمِلْ عَلَيْهِ يَلْهَثْ أَوْ تَتْرُكْهُ يَلْهَثْ ذَلِكَ مَثَلُ الْقَوْمِ الَّذِينَ كَذَّبُوا بِآيَاتِنَا فَاقْصُصِ الْقَصَصَ لَعَلَّهُمْ يَتَفَكَّرُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٧٦</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D8%B9%D8%B1%D8%A7%D9%81" title="سورة الأعراف">الأعراف</a>:176]</small></span>   - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%83%D9%87%D9%81" title="سورة الكهف">سورة الكهف</a> ( 22 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿سَيَقُولُونَ ثَلَاثَةٌ رَابِعُهُمْ كَلْبُهُمْ وَيَقُولُونَ خَمْسَةٌ سَادِسُهُمْ كَلْبُهُمْ رَجْمًا بِالْغَيْبِ وَيَقُولُونَ سَبْعَةٌ وَثَامِنُهُمْ كَلْبُهُمْ قُلْ رَبِّي أَعْلَمُ بِعِدَّتِهِمْ مَا يَعْلَمُهُمْ إِلَّا قَلِيلٌ فَلَا تُمَارِ فِيهِمْ إِلَّا مِرَاءً ظَاهِرًا وَلَا تَسْتَفْتِ فِيهِمْ مِنْهُمْ أَحَدًا <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٢٢</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%83%D9%87%D9%81" title="سورة الكهف">الكهف</a>:22]</small></span></li>
<li><a href="/wiki/%D9%85%D8%A7%D8%B9%D8%B2" title="ماعز">الماعز</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A7%D9%86%D8%B9%D8%A7%D9%85" class="mw-redirect" title="سورة الانعام">سورة الانعام</a> ( 143 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿ثَمَانِيَةَ أَزْوَاجٍ مِنَ الضَّأْنِ اثْنَيْنِ وَمِنَ الْمَعْزِ اثْنَيْنِ قُلْ آلذَّكَرَيْنِ حَرَّمَ أَمِ الْأُنْثَيَيْنِ أَمَّا اشْتَمَلَتْ عَلَيْهِ أَرْحَامُ الْأُنْثَيَيْنِ نَبِّئُونِي بِعِلْمٍ إِنْ كُنْتُمْ صَادِقِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D8%A3%D9%86%D8%B9%D8%A7%D9%85" title="سورة الأنعام">الأنعام</a>:143]</small></span></li>
<li><a href="/wiki/%D9%86%D8%AD%D9%84%D8%A9" class="mw-redirect" title="نحلة">النحلة</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">سورة النحل</a> ( 68 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَأَوْحَى رَبُّكَ إِلَى النَّحْلِ أَنِ اتَّخِذِي مِنَ الْجِبَالِ بُيُوتًا وَمِنَ الشَّجَرِ وَمِمَّا يَعْرِشُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٦٨</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D8%AD%D9%84" title="سورة النحل">النحل</a>:68]</small></span></li>
<li><a href="/wiki/%D9%86%D9%85%D9%84" title="نمل">النمل</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D9%85%D9%84" title="سورة النمل">سورة النمل</a> ( 18 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿حَتَّى إِذَا أَتَوْا عَلَى وَادِ النَّمْلِ قَالَتْ نَمْلَةٌ يَا أَيُّهَا النَّمْلُ ادْخُلُوا مَسَاكِنَكُمْ لَا يَحْطِمَنَّكُمْ سُلَيْمَانُ وَجُنُودُهُ وَهُمْ لَا يَشْعُرُونَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٨</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D9%85%D9%84" title="سورة النمل">النمل</a>:18]</small></span></li>
<li><a href="/wiki/%D9%87%D8%AF%D9%87%D8%AF" title="هدهد">الهدهد</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D9%85%D9%84" title="سورة النمل">سورة النمل</a> ( 20 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿وَتَفَقَّدَ الطَّيْرَ فَقَالَ مَا لِيَ لَا أَرَى الْهُدْهُدَ أَمْ كَانَ مِنَ الْغَائِبِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٢٠</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%86%D9%85%D9%84" title="سورة النمل">النمل</a>:20]</small></span></li>
<li><a href="/wiki/%D8%A7%D9%84%D9%81%D9%8A%D9%84" class="mw-redirect" title="الفيل">الفيل</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%81%D9%8A%D9%84" title="سورة الفيل">سورة الفيل</a> ( 1 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿أَلَمْ تَرَ كَيْفَ فَعَلَ رَبُّكَ بِأَصْحَابِ الْفِيلِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%81%D9%8A%D9%84" title="سورة الفيل">الفيل</a>:1]</small></span></li>
<li><a href="/wiki/%D8%A3%D8%B3%D8%AF" title="أسد">السبع</a>  - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">سورة المائدة</a> ( 3 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿حُرِّمَتْ عَلَيْكُمُ الْمَيْتَةُ وَالدَّمُ وَلَحْمُ الْخِنْزِيرِ وَمَا أُهِلَّ لِغَيْرِ اللَّهِ بِهِ وَالْمُنْخَنِقَةُ وَالْمَوْقُوذَةُ وَالْمُتَرَدِّيَةُ وَالنَّطِيحَةُ وَمَا أَكَلَ السَّبُعُ إِلَّا مَا ذَكَّيْتُمْ وَمَا ذُبِحَ عَلَى النُّصُبِ وَأَنْ تَسْتَقْسِمُوا بِالْأَزْلَامِ ذَلِكُمْ فِسْقٌ الْيَوْمَ يَئِسَ الَّذِينَ كَفَرُوا مِنْ دِينِكُمْ فَلَا تَخْشَوْهُمْ وَاخْشَوْنِ الْيَوْمَ أَكْمَلْتُ لَكُمْ دِينَكُمْ وَأَتْمَمْتُ عَلَيْكُمْ نِعْمَتِي وَرَضِيتُ لَكُمُ الْإِسْلَامَ دِينًا فَمَنِ اضْطُرَّ فِي مَخْمَصَةٍ غَيْرَ مُتَجَانِفٍ لِإِثْمٍ فَإِنَّ اللَّهَ غَفُورٌ رَحِيمٌ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٣</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">المائدة</a>:3]</small></span></li>
<li><a href="/wiki/%D8%BA%D8%B1%D8%A7%D8%A8" title="غراب">الغراب</a> - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">سورة المائدة</a> ( 31 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿فَبَعَثَ اللَّهُ غُرَابًا يَبْحَثُ فِي الْأَرْضِ لِيُرِيَهُ كَيْفَ يُوَارِي سَوْأَةَ أَخِيهِ قَالَ يَا وَيْلَتَا أَعَجَزْتُ أَنْ أَكُونَ مِثْلَ هَذَا الْغُرَابِ فَأُوَارِيَ سَوْأَةَ أَخِي فَأَصْبَحَ مِنَ النَّادِمِينَ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">٣١</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%A7%D9%84%D9%85%D8%A7%D8%A6%D8%AF%D8%A9" title="سورة المائدة">المائدة</a>:31]</small></span></li>
<li><a href="/wiki/%D8%AF%D8%A7%D8%A8%D8%A9_%D8%A7%D9%84%D8%A3%D8%B1%D8%B6" title="دابة الأرض">دابة الأرض</a> - <a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%B3%D8%A8%D8%A3" title="سورة سبأ">سورة سبأ</a> ( 14 )&nbsp;: <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56526893"><span class="quran">﴿فَلَمَّا قَضَيْنَا عَلَيْهِ الْمَوْتَ مَا دَلَّهُمْ عَلَى مَوْتِهِ إِلَّا دَابَّةُ الْأَرْضِ تَأْكُلُ مِنْسَأَتَهُ فَلَمَّا خَرَّ تَبَيَّنَتِ الْجِنُّ أَنْ لَوْ كَانُوا يَعْلَمُونَ الْغَيْبَ مَا لَبِثُوا فِي الْعَذَابِ الْمُهِينِ <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r56889424"><span class="end-of-aya"><a href="/wiki/%D9%85%D9%84%D9%81:AyaEnd.svg" class="image"><img alt="۝" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/20px-AyaEnd.svg.png" decoding="async" width="20" height="26" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/30px-AyaEnd.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8b/AyaEnd.svg/40px-AyaEnd.svg.png 2x" data-file-width="512" data-file-height="669"></a><span class="aya-num">١٤</span></span>﴾&nbsp;<small class="quran-cite">[<a href="/wiki/%D8%B3%D9%88%D8%B1%D8%A9_%D8%B3%D8%A8%D8%A3" title="سورة سبأ">سبأ</a>:14]</small></span></li></ol>"""


soup = BeautifulSoup(list_of_animals, "html.parser")

animals = []
for li in soup.find_all("li"):
    animal_name = li.text.split(" - ")[0].split(":")[-1].strip()
    if animal_name.startswith("ال"):
        animal_name = animal_name[2:]
    animals.append(animal_name)

with open("data/animals.txt", "w") as f:
    f.write("\n".join(animals))
