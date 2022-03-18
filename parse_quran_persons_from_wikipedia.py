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
