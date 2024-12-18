## Задание
Предложите свой дизайн для системы глобального дейтингового сервиса (аналога Tinder, Bumble, Badoo, etc.). Сразу обозначим, что в максимально приближенной к действительности постановке задачи, это будет мобильное приложение и веб-интерфейс, представленные (и соответственно локализованные) на рынке практически всех стран мира, c ± 100M MAU (Monthly active users) и ±10M DAU (Daily active users)

Нужно спроектировать не только все фичи из первого и второго уровней, но также, добавить в картину то, насколько много данных и ивентов на самом деле генерируют наши юзеры, и как мы это все собираем, храним и используем. Согласно техническому блогу компании Tinder, сейчас их юзеры генерируют порядка 1 млн. ивентов в секунду (или порядка 86 млрд. в сутки). Это данные о действиях юзеров в приложении, плюс различная телеметрия и метрики, которые система получает по сети, обрабатывает и затем в некоторой форме хранит и использует в своих сервисах рекомендаций, аналитики, безопасности и многом другом.

Взять в скоуп требование о том, что мы собираем и храним 1M events/sec пользовательских событий. Продумать, как будет выглядеть масштабирующийся ETL для такого количества событий, в каких базах и как долго все эти данные будут храниться.
Добавить в конечный дизайн несколько ML-компонент, делающих предсказания на основе собираемых данных: например, сервис, который сортирует отправляемые пользователям непросмотренные анкеты, увеличивая вероятность совпадений; сервис рекомендации анкет или сервис промо-пушей, стимулирующих пользовательскую активность.

Небольшая памятка-чеклист, напоминающая о всех важных шагах процесса по дизайну системы, а так же некоторые нефункциональные области, требования в которых можно было бы дополнительно поискать и проработать в конечном дизайне для улучшения его полноты.

1. Начать со сбора функциональных и нефункциональных требований, после формирования начального списка проанализировать его на предмет противоречий или важных производных требований.
2. Прикинуть основные паттерны активности юзеров и сделать оценки нагрузки.
3. На основе предполагаемой нагрузки и информации о юзерах сделать оценки расчета затрат на трафик, хранение данных и вычислительные мощности.
4. Рассмотреть высокоуровневый дизайн который удовлетворяет всем основным функциональным требованиям.
Далее в случае большой системы с разнородным функционалом можно выделить важные подсистемы и уже рассматривать их в деталях модульно (например, рассмотреть отдельно базовые апи, отдельно реал-тайм подсистемы вроде чата, отдельно - геолокационный поиск)
5. Рассмотреть особенности хранения данных: обоснованность выбора баз и их репликацию.
6. Рассмотреть методы повышения надежности и отзывчивости.
7. При проектировании любой большой системы, состоящей из множества сервисов важно не забыть уделить внимание ее точкам входа. Скорей всего, там потребуются балансировщики и умные роутеры/гейтвеи, которые будут закрывать от внешнего мира всю вашу инфраструктуру и решать массу задач по обеспечению безопасности и надежности решения.
8. Подумать об инфраструктуре сбора и хранения данных, логов и телеметрии.
9. Подумать о наличии инфраструктуры для решения задач машинного обучения.
10. В качестве дополнительного технического скоупа можно добавить требования по безопасности, и спроектировать подсистемы, связанные с безопасностью: управление ключами шифрования, идентификация юзеров, защита данных при хранении и передаче, хранение и проверка прав на доступ и т.д.
11. Также желательно сопроводить ход ваших рассуждений небольшими комментариями в сносках к блоками на схеме или в отдельном текстовом блоке.