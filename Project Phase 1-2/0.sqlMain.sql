
WITH BadgeCounts AS (SELECT userid,
                            sum(case when class = 1 then 1 else 0 end) AS gold,
                            sum(case when class = 2 then 1 else 0 end) AS silver,
                            sum(case when class = 3 then 1 else 0 end) AS bronze
                     FROM badges
                     GROUP BY userid)

SELECT DISTINCT question.Id                                                                        AS QuestionId,
                question.Title                                                                     AS QuestionTitle,
                question.Body                                                                      AS QuestionBody,
                question.Tags                                                                      AS QuestionTags,
                len(question.Body)                                                                 AS QuestionBodyLength,
                users.reputation                                                                   AS UserReputation,
                badge_counts.gold                                                                  AS GoldBadges,
                badge_counts.silver                                                                AS SilverBadges,
                badge_counts.bronze                                                                As BronzeBadges,
                question.ViewCount                                                                 AS QuestionViewCount,
                question.FavoriteCount                                                             AS QuestionFavoriteCount,
                users.UpVotes                                                                      As UserUpVotes,
                question.AnswerCount                                                               AS AnswerCount,
                question.Score                                                                     AS QuestionScore,
                question.CreationDate                                                              AS QuestionCreationDate,
                (SELECT TOP 1 posts.CreationDate
                 FROM posts
                 WHERE question.Id = posts.ParentId
                 ORDER BY posts.CreationDate)                                                      AS FirstAnswerCreationDate,
                (SELECT TOP 1 posts.CreationDate
                 FROM posts
                 WHERE question.AcceptedAnswerId = posts.Id)                                       AS AcceptedAnswerCreationDate,
                DATEDIFF(day, question.CreationDate, (SELECT TOP 1 posts.CreationDate
                                                      FROM posts
                                                      WHERE question.Id = posts.ParentId
                                                      ORDER BY posts.CreationDate))                as FirstAnswerIntervalDays,
                DATEDIFF(day, question.CreationDate, (SELECT TOP 1 posts.CreationDate
                                                      FROM posts
                                                      WHERE question.AcceptedAnswerId = posts.Id)) as AcceptedAnswerIntervalDays

FROM Posts question
         LEFT JOIN
     PostTags post_tag ON question.Id = post_tag.PostId
         LEFT JOIN
     Tags tags ON post_tag.TagId = tags.Id
         JOIN
     BadgeCounts badge_counts ON question.owneruserid = badge_counts.userid
         JOIN
     users users ON question.owneruserid = users.id
         JOIN
     Comments comments ON question.Id = comments.PostId

WHERE question.CreationDate >= '2018-01-01'
  AND question.CreationDate <= '2020-12-31'
  AND question.PostTypeId = 1
  AND question.tags LIKE '%sql%'
GROUP BY question.Id,
         question.Title,
         question.Body,
         question.Tags,
         users.reputation,
         badge_counts.gold,
         badge_counts.silver,
         badge_counts.bronze,
         question.ViewCount,
         question.FavoriteCount,
         question.AnswerCount,
         question.Score,
         question.CreationDate,
         question.AcceptedAnswerId,
         users.UpVotes
ORDER BY question.Score DESC
OFFSET 0 ROWS FETCH NEXT 1000 ROWS ONLY;